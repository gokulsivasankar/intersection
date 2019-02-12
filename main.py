# Date Feb 6, 2019
# 
# Main file for the intersection environment simulator

import math

import get_params
import traff
import numpy as np
import matplotlib.pyplot as plt

params = get_params.get_params()


for episode in range(1,params.max_episode+1):
    # traffic initialization
    traffic = traff.initial()

    # Ego car
    x_car = 0.5*params.w_lane
    y_car = -4*params.w_lane
    orientation_car = math.pi/2
    v_car = params.v_nominal
    target_car = 2

    # traffic update
    traffic = traff.update(traffic,x_car,y_car,orientation_car,v_car,target_car)

    Level_ratio = np.matrix('0.1, 0.6, 0.3;0.1, 0.6, 0.3') # Initial guess for the level ratio (0 1 2)

    for id in range(2,params.num_cars+1):
        if id==2:
            x_car = -0.5*params.w_lane
            y_car = 4*params.w_lane
            orientation_car = -math.pi/2
            v_car = params.v_nominal
            target_car = 1
            
    # traffic update
    traffic = traff.update(traffic,x_car,y_car,orientation_car,v_car,target_car)
    initial_state = np.block([[traffic.x],[traffic.y],[traffic.orientation],[traffic.v],[traffic.target]])
    # action space 
    action_space = np.matrix('1, 2, 3, 4, 5, 6')
    # 1: maintain 2: turn left 3: turn right 4:accelerate 5: decelerate 6: hard brake
    
    X_old = initial_state
    
    a=1;
    for step in range(1,51):
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
        # I cannot find the corresponding function of matlab's cell, but this list provides the similar function.        
        L0_action_id = [{}]*params.num_cars
        L0_Q_value = [{}]*params.num_cars
        for car_id in range(1,params.num_cars+1):
            a = DecisionTree_L0(X_old, car_id, action_space, params.t_step_DT) # Call the decision tree function
            print(a)

            print()
            print()
            
