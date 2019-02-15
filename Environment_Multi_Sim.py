# -*- coding: utf-8 -*-
import numpy as np

def Environment_Multi_Sim(X_old, action_id, t_step):
    
    # the selected car moves based on the action
    # 1:maintian  2:turn left  3:turn right  4:accelerate  5:decelerate  6:hard brake
    size_state = X_old.size  # maybe doesn't need, 
    # we don't need to create the 3d matrix...
    X_new =zeros((action_id[0].size))
    for step in range(0, action_id[0].size):
        size_action_cell = action_id.size
        for car_id in range(0, size_action_cell[0]):
            X_old = Motion_Update(X_old, car_id, action_id[car_id][step], t_step)
        X_new[step]=X_old     
        for car_id in range(0, size_action_cell[0]):
            #### continue
    return X_new, R