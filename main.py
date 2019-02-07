# Date Feb 6, 2019
# 
# Main file for the intersection environment simulator

import math

import get_params
import traff
import numpy as np

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

    Level_ratio = np.matrix('0.1, 0.6, 0.3;0.1, 0.6, 0.3')

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

    print(np.shape(initial_state))
    print((initial_state[0,1]))
        