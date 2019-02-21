# Date Feb 6, 2019
# 
# Main file for the intersection environment simulator

import math
import get_params
import traff
import numpy as np
import matplotlib.pyplot as plt

params = get_params.get_params()    # call a user-defined function
w_lane = params.w_lane
v_nominal = params.v_nominal
num_cars = num_cars
l_car = params.l_car
w_car = params.w_car
max_episode = params.max_episode
t_step_DT = params.t_step_DT

for episode in range(1,params.max_episode+1):    # simulation will be runned 1 time
    #plt.pyplot.close               # corresponds to close all of matlab, but need to check
    # Traffic initialization
    traffic = traff.initial()       # call a user-defined function, x,y, v, target traffic

    # Ego car 
    x_car = 0.5*w_lane       # ego vehicles' initial x position
    y_car = -4*w_lane        # ego vehicles' initial y position
    orientation_car = math.pi/2     # ego vehicles' initial heading angle (in the paper, yaw angle)
    v_car = v_nominal        # ego vehicles' initial speed
    target_car = 2                  # target car is 2(opponent)

    # traffic update
    traffic = traff.update(traffic, x_car, y_car, orientation_car, v_car, target_car)
    # Initial guess for the level ratio (0 1 2)
    Level_ratio = np.array([[0.1, 0.6, 0.3], [0.1, 0.6, 0.3]]) 

    for id in range(2, num_cars+1):
        if id==2:
            x_car = -0.5*w_lane    # opponent vehicles' initial x position
            y_car = 4*w_lane       # opponent vehicles' initial y position
            orientation_car = -math.pi/2  # opponent vehicles' initial heading angle
            v_car = v_nominal      # opponent vehicles' initial speed
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
        L0_action_id = [ []*num_cars, []*1 ]
        L0_Q_value = [ []*num_cars, []*1 ]
        for car_id in range(0, num_cars):
            L0_Q_value[car_id], L0_action_id[car_id] = DecisionTree_L0(X_old, car_id, action_space, t_step_DT) # Call the decision tree function
           
        X_pseudo_L0 = Environment_Multi(X_old, L0_action_id, t_step_DT)
        
        X_pseudo_L0_Id = [ []*num_cars, []*1 ]
        for car_id in range(0, num_cars):
            X_pseudo_L0_Id[car_id] = X_pseudo_L0
            for pre_step in range(0, L0_action_id[0].size):
                X_pseudo_L0_Id[car_id][pre_step, :, car_id] = X_old[:, car_id]
                
        # L-1
        L1_action_id = [ []*num_cars, []*1 ]
        L1_Q_value = [ []*num_cars, []*1 ]
        for car_id in range(0, num_cars):
            L1_Q_value[car_id], L1_action_id[car_id] = DecisionTree_L1(X_pseudo_L0_Id[car_id], car_id, action_space, t_step_DT) # Call the decision tree function
           
        X_pseudo_L1 = Environment_Multi(X_old, L1_action_id, t_step_DT)
        
        X_pseudo_L1_Id = [ []*num_cars, []*1 ]
        for car_id in range(0, num_cars):
            X_pseudo_L1_Id[car_id] = X_pseudo_L1
            for pre_step in range(0, L1_action_id[0].size):
                X_pseudo_L1_Id[car_id][pre_step, :, car_id] = X_old[:, car_id]
        
        # L-2
        L2_action_id = [ []*num_cars, []*1 ]
        L2_Q_value = [ []*num_cars, []*1 ]
        for car_id in range(0, num_cars):
            L2_Q_value[car_id], L2_action_id[car_id] = DecisionTree_L1(X_pseudo_L1_Id[car_id], car_id, action_space, t_step_DT) # Call the decision tree function
           
        X_pseudo_L2 = Environment_Multi(X_old, L2_action_id, t_step_DT)
        
        X_pseudo_L2_Id = [ []*num_cars, []*1 ]
        for car_id in range(0, num_cars):
            X_pseudo_L2_Id[car_id] = X_pseudo_L2
            for pre_step in range(0, L2_action_id[0].size):
                X_pseudo_L2_Id[car_id][pre_step, :, car_id] = X_old[:, car_id]
        
        # D-2
        D2_action_id = [ []*num_cars, []*1 ]
        D2_Q_value = [ []*num_cars, []*1 ]
        for car_id in range(0, num_cars):
            D2_Q_value[car_id] = 1/2*L1_Q_value[car_id] + 1/2 * L2_Q_value[car_id]
            D2_action_id[car_id] = D2_Q_value[car_id].index(max(D2_Q_value[car_id]))
            
        # L-3
        L3_action_id = [ []*num_cars, []*1 ]
        L3_Q_value = [ []*num_cars, []*1 ]
        for car_id in range(0, num_cars):
            L3_Q_value[car_id], L3_action_id[car_id] = DecisionTree_L1(X_pseudo_L2_Id[car_id], car_id, action_space, t_step_DT) # Call the decision tree function
           
        X_pseudo_L3 = Environment_Multi(X_old, L3_action_id, t_step_DT)
        
        X_pseudo_L3_Id = [ []*num_cars, []*1 ]
        for car_id in range(0, num_cars):
            X_pseudo_L3_Id[car_id] = X_pseudo_L3
            for pre_step in range(0, L3_action_id[0].size):
                X_pseudo_L3_Id[car_id][pre_step, :, car_id] = X_old[:, car_id]
        # D-3
        D3_action_id = [ []*num_cars, []*1 ]
        D3_Q_value = [ []*num_cars, []*1 ]
        for car_id in range(0, num_cars):
            D3_Q_value[car_id] = Level_ratio[car_id, 0]*L1_Q_value[car_id] 
                                + Level_ratio[car_id, 1]*L2_Q_value[car_id]
                                + Level_ratio[car_id, 2]*L3_Q_value[car_id]
            D3_action_id[car_id] = D3_Q_value[car_id].index(max(D3_Q_value[car_id]))
        
        Action_id = [ []*num_cars, []*1 ]
        Action_id[0] = L2_action_id[0][0]
        Action_id[1] = L2_action_id[1][0]
        
        # Level estimation update
        for car_id in range(0, num_cars):
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




        # Gokul's part
    car_id = 1
    color = ['b','r','m','g']

    # Animation plot

    fig, ax = plt.subplots()
    fig.clf()
        
    #     x = [-30 -30 0 0]
    #     y = [-30 30 30 -30]
    #     p = patch(x,y,'r')
    #     set(p,'FaceAlpha',0.1)
        
    RoadBound = np.matrix([
        [w_lane,-6*w_lane],
        [w_lane,-2*w_lane],
        [2*w_lane,-w_lane],
        [6*w_lane,-w_lane],
        [6*w_lane,-6*w_lane],
        [w_lane,6*w_lane],
        [w_lane,2*w_lane],
        [2*w_lane,w_lane],
        [6*w_lane,w_lane],
        [6*w_lane,6*w_lane],
        [-w_lane,-6*w_lane],
        [-w_lane,-2*w_lane],
        [-2*w_lane,-w_lane],
        [-6*w_lane,-w_lane],
        [-6*w_lane,-6*w_lane],
        [-w_lane,6*w_lane],
        [-w_lane,2*w_lane],
        [-2*w_lane,w_lane],
        [-6*w_lane,w_lane],
        [-6*w_lane,6*w_lane]])
        
        # Indices taken care
    ax.fill(np.squeeze(RoadBound[0:4,0]),np.squeeze(RoadBound[0:4,1]),color='0.75',LineWidth = 2)
    ax.fill(np.squeeze(RoadBound[5:9,0]),np.squeeze(RoadBound[5:9,1]),color='0.75',LineWidth = 2)
    ax.fill(np.squeeze(RoadBound[10:14,0]),np.squeeze(RoadBound[10:14,1]),color='0.75',LineWidth = 2)
    ax.fill(np.squeeze(RoadBound[15:16,0]),np.squeeze(RoadBound[15:16,1]),color='0.75',LineWidth = 2)

    RoadMid = np.matrix([
        [2*w_lane, 0],
        [10*w_lane, 0],
        [-2*w_lane, 0],
        [-10*w_lane, 0],
        [0, -2*w_lane],
        [0, -10*w_lane],
        [0, 2*w_lane],
        [0, 10*w_lane]])

    plt.plot(np.squeeze(RoadMid[0:1,0]),np.squeeze(RoadMid[0:1,1]),color=(1,0.5,0),LineWidth = 3)
    plt.plot(np.squeeze(RoadMid[2:3,0]),np.squeeze(RoadMid[2:3,1]),color=(1,0.5,0),LineWidth = 3)
    plt.plot(np.squeeze(RoadMid[4:5,0]),np.squeeze(RoadMid[4:5,1]),color=(1,0.5,0),LineWidth = 3)
    plt.plot(np.squeeze(RoadMid[6:7,0]),np.squeeze(RoadMid[6:7,1]),color=(1,0.5,0),LineWidth = 3)

    # Indices please!
    Ego_rectangle = np.matrix(
        [[X_old[1,car_id]-l_car/2*math.cos(X_old[3,car_id])-w_car/2*math.sin(X_old[3,car_id]),X_old[2,car_id]-l_car/2*math.sin(X_old[3,car_id])+w_car/2*math.cos(X_old[3,car_id])],
        [X_old[1,car_id]-l_car/2*math.cos(X_old[3,car_id])+w_car/2*math.sin(X_old[3,car_id]), X_old[2,car_id]-l_car/2*math.sin(X_old[3,car_id])-w_car/2*math.cos(X_old[3,car_id])],
        [X_old[1,car_id]+l_car/2*math.cos(X_old[3,car_id])-w_car/2*math.sin(X_old[3,car_id]), X_old[2,car_id]+l_car/2*math.sin(X_old[3,car_id])+w_car/2*math.cos(X_old[3,car_id])],
        [X_old[1,car_id]+l_car/2*math.cos(X_old[3,car_id])+w_car/2*math.sin(X_old[3,car_id]), X_old[2,car_id]+l_car/2*math.sin(X_old[3,car_id])-w_car/2*math.cos(X_old[3,car_id])],
        [X_old[1,car_id]+(l_car/2-1)*math.cos(X_old[3,car_id])-w_car/2*math.sin(X_old[3,car_id]), X_old[2,car_id]+(l_car/2-1)*math.sin(X_old[3,car_id])+w_car/2*math.cos(X_old[3,car_id])],
        [X_old[1,car_id]+(l_car/2-1)*math.cos(X_old[3,car_id])+w_car/2*math.sin(X_old[3,car_id]), X_old[2,car_id]+(l_car/2-1)*math.sin(X_old[3,car_id])-w_car/2*math.cos(X_old[3,car_id])]])

    # Indices taken care
    plt.plot(np.squeeze(Ego_rectangle[0:1,0]),np.squeeze(Ego_rectangle[0:1,1]),color=color[car_id],LineWidth=3,linestyle='-')
    plt.plot([Ego_rectangle[0,0],Ego_rectangle[2,0]],[Ego_rectangle[0,1],Ego_rectangle[2,1]],color=color[car_id],LineWidth=3,linestyle='-')
    plt.plot([Ego_rectangle[2,0],Ego_rectangle[3,0]],[Ego_rectangle[2,1],Ego_rectangle[3,1]],color=color[car_id],LineWidth=3,linestyle='-')
    plt.plot([Ego_rectangle[1,0],Ego_rectangle[3,0]],[Ego_rectangle[1,1],Ego_rectangle[3,1]],color=color[car_id],LineWidth=3,linestyle='-')
    plt.plot([Ego_rectangle[4,0],Ego_rectangle[5,0]],[Ego_rectangle[4,1],Ego_rectangle[5,1]],color=color[car_id],LineWidth=3,linestyle='-')

        
    for id in range(1,len(X_old[1,:])):
        if id!=car_id:

            Other_rectangle = np.matrix(
                [[X_old[1,id]-l_car/2*math.cos(X_old[3,id])-w_car/2*math.sin(X_old[3,id]),X_old[2,id]-l_car/2*math.sin(X_old[3,id])+w_car/2*math.cos(X_old[3,id])],
                [X_old[1,id]-l_car/2*math.cos(X_old[3,id])+w_car/2*math.sin(X_old[3,id]), X_old[2,id]-l_car/2*math.sin(X_old[3,id])-w_car/2*math.cos(X_old[3,id])],
                [X_old[1,id]+l_car/2*math.cos(X_old[3,id])-w_car/2*math.sin(X_old[3,id]), X_old[2,id]+l_car/2*math.sin(X_old[3,id])+w_car/2*math.cos(X_old[3,id])],
                [X_old[1,id]+l_car/2*math.cos(X_old[3,id])+w_car/2*math.sin(X_old[3,id]), X_old[2,id]+l_car/2*math.sin(X_old[3,id])-w_car/2*math.cos(X_old[3,id])],
                [X_old[1,id]+(l_car/2-1)*math.cos(X_old[3,id])-w_car/2*math.sin(X_old[3,id]), X_old[2,id]+(l_car/2-1)*math.sin(X_old[3,id])+w_car/2*math.cos(X_old[3,id])],
                [X_old[1,id]+(l_car/2-1)*math.cos(X_old[3,id])+w_car/2*math.sin(X_old[3,id]), X_old[2,id]+(l_car/2-1)*math.sin(X_old[3,id])-w_car/2*math.cos(X_old[3,id])]])


            plt.plot(np.squeeze(Other_rectangle[0:1,0]),np.squeeze(Other_rectangle[0:1,1]),color=color[id],LineWidth=3,linestyle='-')
            plt.plot([Other_rectangle[0,0],Other_rectangle[2,0]],[Other_rectangle[0,1],Other_rectangle[2,1]],color=color[id],LineWidth=3,linestyle='-')
            plt.plot([Other_rectangle[2,0],Other_rectangle[3,0]],[Other_rectangle[2,1],Other_rectangle[3,1]],color=color[id],LineWidth=3,linestyle='-')
            plt.plot([Other_rectangle[1,0],Other_rectangle[3,0]],[Other_rectangle[1,1],Other_rectangle[3,1]],color=color[id],LineWidth=3,linestyle='-')
            plt.plot([Other_rectangle[4,0],Other_rectangle[5,0]],[Other_rectangle[4,1],Ego_rectangle[5,1]],color=color[id],LineWidth=3,linestyle='-')



    ax.annotate(
        'v='+str(X_old[3,0])+'m/s', xytext=(3, 1.5))    
    ax.annotate(
        'v='+str(X_old[3,1])+'m/s', xytext=(1, 0))
            
    ax.set_xlim(-6*w_lane, 6*w_lane)
    ax.set_ylim(-6*w_lane, 6*w_lane)
        
        # convert this animation part
        #frame = getframe(gcf)
        #im = frame2im(frame)
        #[imind,cm] = rgb2ind(im,256)
        #if step==1
        #    imwrite(imind,cm,outfile,'gif','DelayTime',0,'loopcount',inf)
        #else
        #    imwrite(imind,cm,outfile,'gif','DelayTime',0,'writemode','append')
        #end
        #pause(0.001)
        
    fig = plt.subplots()
    plt.plot(step,R[0],color='b',marker='.',markersize=16)
    plt.plot(step,R[1],color='r',marker='.',markersize=16)
        
    if sum(complete_flag[episode,:])==num_cars:
        break
        
    if(R[0]<-5000 or R[1]<-5000):
        break
        

complete_ratio = sum(complete_flag[:,1]*complete_flag[:,2])/max_episode

#fig = plt.subplots()
#plt.plot(range(1:len(R_history[1,:,end])]*t_step_Sim,R_history(1,:,end),'b-','LineWidth',3)
#plot([1:1:length(R_history(2,:,end))]*t_step_Sim,R_history(2,:,end),'r-','LineWidth',3)
#xlabel('t [s]')
#ylabel('R')

#figure(21) hold on
#plot([1:1:length(reshape(Level_ratio_history(1,1,:,end),[],1))]*t_step_Sim,reshape(Level_ratio_history(1,1,:,end),[],1),'b-','LineWidth',3)
#plot([1:1:length(reshape(Level_ratio_history(1,1,:,end),[],1))]*t_step_Sim,reshape(Level_ratio_history(1,2,:,end),[],1),'r-','LineWidth',3)
#plot([1:1:length(reshape(Level_ratio_history(1,1,:,end),[],1))]*t_step_Sim,reshape(Level_ratio_history(1,3,:,end),[],1),'g-','LineWidth',3)
#title('Car 1''s estimate on car 2')
#xlabel('t [s]')
#ylabel('Ratio')

#figure(22) hold on
#plot([1:1:length(reshape(Level_ratio_history(1,1,:,end),[],1))]*t_step_Sim,reshape(Level_ratio_history(2,1,:,end),[],1),'b-','LineWidth',3)
#plot([1:1:length(reshape(Level_ratio_history(1,1,:,end),[],1))]*t_step_Sim,reshape(Level_ratio_history(2,2,:,end),[],1),'r-','LineWidth',3)
#plot([1:1:length(reshape(Level_ratio_history(1,1,:,end),[],1))]*t_step_Sim,reshape(Level_ratio_history(2,3,:,end),[],1),'g-','LineWidth',3)
#title('Car 2''s estimate on car 1')
#xlabel('t [s]')
#ylabel('Ratio')

#save Q_table
#np.savetxt('Q_table.csv', ?which variables to save?, delimiter=',', fmt='%d')