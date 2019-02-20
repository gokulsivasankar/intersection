# Date Feb 6, 2019
# 
# Main file for the intersection environment simulator

import math
import get_params
import traff
import numpy as np
import matplotlib.pyplot as plt

params = get_params.get_params()    # call a user-defined function

for episode in range(1,params.max_episode+1):    # simulation will be runned 1 time
    #plt.pyplot.close               # corresponds to close all of matlab, but need to check
    # Traffic initialization
    traffic = traff.initial()       # call a user-defined function, x,y, v, target traffic

    # Ego car 
    x_car = 0.5*params.w_lane       # ego vehicles' initial x position
    y_car = -4*params.w_lane        # ego vehicles' initial y position
    orientation_car = math.pi/2     # ego vehicles' initial heading angle (in the paper, yaw angle)
    v_car = params.v_nominal        # ego vehicles' initial speed
    target_car = 2                  # target car is 2(opponent)

    # traffic update
    traffic = traff.update(traffic, x_car, y_car, orientation_car, v_car, target_car)
    # Initial guess for the level ratio (0 1 2)
    Level_ratio = np.array([[0.1, 0.6, 0.3], [0.1, 0.6, 0.3]]) 

    for id in range(2, params.num_cars+1):
        if id==2:
            x_car = -0.5*params.w_lane    # opponent vehicles' initial x position
            y_car = 4*params.w_lane       # opponent vehicles' initial y position
            orientation_car = -math.pi/2  # opponent vehicles' initial heading angle
            v_car = params.v_nominal      # opponent vehicles' initial speed
            target_car = 1                # target car is 1 (ego?)
            
    # traffic update
    traffic = traff.update(traffic, x_car, y_car, orientation_car, v_car,target_car)
    initial_state = np.block ([[traffic.x], [traffic.y], [traffic.orientation],
                               [traffic.v_car], [traffic.target_car]])
    # action space 
    action_space = np.array([[1,2,3,4,5,6]])
    # 1: maintain 2: turn left 3: turn right 4:accelerate 5: decelerate 6: hard brake
    
    X_old = initial_state
    step_size = 50
    for step in range(0, step_size):
        #fprintf('step = %i\n', step) ?
        
        plt.figure(1)
        plt.plot(step, Level_ratio[0,0], 'b.')
        plt.plot(step, Level_ratio[0,1], 'r.')
        plt.plot(step, Level_ratio[0,2], 'g.')
        plt.title('Car1')
        
        plt.figure(2)
        plt.plot(step, Level_ratio[1,0], 'b.')
        plt.plot(step, Level_ratio[1,1], 'r.')
        plt.plot(step, Level_ratio[1,2], 'g.')
        plt.title('Car2')
       
        # L-0
        L0_action_id = [ []*params.num_cars, []*1 ]
        L0_Q_value = [ []*params.num_cars, []*1 ]
        for car_id in range(0, params.num_cars):
            L0_Q_value[car_id], L0_action_id[car_id] = DecisionTree_L0(X_old, car_id, action_space, params.t_step_DT) # Call the decision tree function
           
        X_pseudo_L0 = Environment_Multi(X_old, L0_action_id, t_step_DT)
        
        X_pseudo_L0_Id = [ []*params.num_cars, []*1 ]
        for car_id in range(0, params.num_cars):
            X_pseudo_L0_Id[car_id] = X_pseudo_L0
            for pre_step in range(0, L0_action_id[0].size):
                X_pseudo_L0_Id[car_id][pre_step, :, car_id] = X_old[:, car_id]
                
        # L-1
        L1_action_id = [ []*params.num_cars, []*1 ]
        L1_Q_value = [ []*params.num_cars, []*1 ]
        for car_id in range(0, params.num_cars):
            L1_Q_value[car_id], L1_action_id[car_id] = DecisionTree_L1(X_pseudo_L0_Id[car_id], car_id, action_space, params.t_step_DT) # Call the decision tree function
           
        X_pseudo_L1 = Environment_Multi(X_old, L1_action_id, t_step_DT)
        
        X_pseudo_L1_Id = [ []*params.num_cars, []*1 ]
        for car_id in range(0, params.num_cars):
            X_pseudo_L1_Id[car_id] = X_pseudo_L1
            for pre_step in range(0, L1_action_id[0].size):
                X_pseudo_L1_Id[car_id][pre_step, :, car_id] = X_old[:, car_id]
        
        # L-2
        L2_action_id = [ []*params.num_cars, []*1 ]
        L2_Q_value = [ []*params.num_cars, []*1 ]
        for car_id in range(0, params.num_cars):
            L2_Q_value[car_id], L2_action_id[car_id] = DecisionTree_L1(X_pseudo_L1_Id[car_id], car_id, action_space, params.t_step_DT) # Call the decision tree function
           
        X_pseudo_L2 = Environment_Multi(X_old, L2_action_id, t_step_DT)
        
        X_pseudo_L2_Id = [ []*params.num_cars, []*1 ]
        for car_id in range(0, params.num_cars):
            X_pseudo_L2_Id[car_id] = X_pseudo_L2
            for pre_step in range(0, L2_action_id[0].size):
                X_pseudo_L2_Id[car_id][pre_step, :, car_id] = X_old[:, car_id]
        
        # D-2
        D2_action_id = [ []*params.num_cars, []*1 ]
        D2_Q_value = [ []*params.num_cars, []*1 ]
        for car_id in range(0, params.num_cars):
            D2_Q_value[car_id] = 1/2*L1_Q_value[car_id] + 1/2 * L2_Q_value[car_id]
            D2_action_id[car_id] = D2_Q_value[car_id].index(max(D2_Q_value[car_id]))
            
        # L-3
        L3_action_id = [ []*params.num_cars, []*1 ]
        L3_Q_value = [ []*params.num_cars, []*1 ]
        for car_id in range(0, params.num_cars):
            L3_Q_value[car_id], L3_action_id[car_id] = DecisionTree_L1(X_pseudo_L2_Id[car_id], car_id, action_space, params.t_step_DT) # Call the decision tree function
           
        X_pseudo_L3 = Environment_Multi(X_old, L3_action_id, t_step_DT)
        
        X_pseudo_L3_Id = [ []*params.num_cars, []*1 ]
        for car_id in range(0, params.num_cars):
            X_pseudo_L3_Id[car_id] = X_pseudo_L3
            for pre_step in range(0, L3_action_id[0].size):
                X_pseudo_L3_Id[car_id][pre_step, :, car_id] = X_old[:, car_id]
        # D-3
        D3_action_id = [ []*params.num_cars, []*1 ]
        D3_Q_value = [ []*params.num_cars, []*1 ]
        for car_id in range(0, params.num_cars):
            D3_Q_value[car_id] = Level_ratio[car_id, 0]*L1_Q_value[car_id] 
                                + Level_ratio[car_id, 1]*L2_Q_value[car_id]
                                + Level_ratio[car_id, 2]*L3_Q_value[car_id]
            D3_action_id[car_id] = D3_Q_value[car_id].index(max(D3_Q_value[car_id]))
        
        Action_id = [ []*params.num_cars, []*1 ]
        Action_id[0] = L2_action_id[0][0]
        Action_id[1] = L2_action_id[1][0]
        
        # Level estimation update
        for car_id in range(0, params.num_cars):
            if L0_action_id[2-car_id][0]==L1_action_id{2-car_id}(0) and L1_action_id[2-car_id](0)==L2_action_id[2-car_id][0]:
            else:
                if Action_id[2-car_id] == L0_action_id[2-car_id][0]:
                    Level_ratio(car_id, 0) = Level_ratio[car_id, 0]+0.5; 
                if Action_id[2-car_id] == L1_action_id[2-car_id][0]:
                    Level_ratio(car_id, 1) = Level_ratio[car_id, 1]+0.5;
                if Action_id[2-car_id] == L2_action_id[2-car_id][0]:
                    Level_ratio(car_id, 2) = Level_ratio[car_id, 2]+0.5;   
            Level_ratio[car_id,:] = Level_ratio[car_id,:]/ sum([Level_ratio[car_id,:]])
        
        Level_ratio_history=zeros( (step_size, np.shape(Level_ratio)[0], np.shape(Level_ratio)[1]) )
        Level_ratio_history[step,:,:] = Level_ratio
        
        # State update
        X_new, R =Environment_Multi_Sim(X_old, Action_id, t_step_Sim)
        X_old = X_new
        
        R_history = np.zeros((2, step_size))
        R_history[np.shape(R)[0], step] = R
        
        car_id = 1
        
        
        
        
        
        
        
        
        
        
        
        
        
        