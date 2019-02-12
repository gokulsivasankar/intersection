# -*- coding: utf-8 -*-
import numpy as np

def DecisionTree_L0(X_old, car_id, action_space, t_step_DT):
    index = [{}]*action_space.size
    for i in range(0,np.size(index)):
        index[i]=1
        
    Q_value = [{}]*action_space.size
    for i in range(0,np.size(Q_value)):
        Q_value[i]=1
        
    action_id = [{}]*action_space.size
    Buffer = [{}]*4
    
    R1_max, R2_max, R3_max, R4_max, R5_max = -1e6, -1e6, -1e6, -1e6, -1e6
    for id_1 in range(0,np.size(Q_value)):
        k=1
        #X_old=Buffer[0,k]
    return Buffer
