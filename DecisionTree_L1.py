# -*- coding: utf-8 -*-
import numpy as np

def DecisionTree_L0(X_pseudo, car_id, action_space, t_step_DT):
    index = [{1}]*action_space.size       
    Q_value = [{-1e6}]*action_space.size
        
    action_id = [{}]*action_space.size
    Buffer = [{}]*4
    Buffer[0] = X_psuedo[0]
    R1_max, R2_max, R3_max, R4_max, R5_max = -1e6, -1e6, -1e6, -1e6
    for id_1 in range(0, action_space.size):
        k=0
        X_old = Buffer[k]
        X_new, R1 = Environment(X_old, car_id, id_1, t_step_DT)
        R1_max = max(R1_max, R1)      
        Buffer[k+1]=X_new[k+1]
        Buffer[k+1][car_id-1]=X_new[car_id-1]
        for id_2 in range(0, action_space.size):
            k=1
            X_old=Buffer[k]
            X_new, R2 = Environment(X_old, car_id, id_2, t_step_DT)
            R2_max = max(R2_max, R2)
            Buffer[k+1]=X_new[k+1]
            Buffer[k+1][car_id-1]=X_new[car_id-1]
            for id_3 in range(0, action_space.size):
                k=2
                X_old=Buffer[k]
                X_new, R3 = Environment(X_old, car_id, id_3, t_step_DT)
                R3_max = max(R3_max, R3)
                Buffer[k+1]=X_new[k+1]
                Buffer[k+1][car_id-1]=X_new[car_id-1]
                for id_4 in range(0, action_space.size):
                    k=3
                    X_old=Buffer[k]
                    X_new, R4 = Environment(X_old, car_id, id_4, t_step_DT)
                    R4_max = max(R4_max, R4)
                   
                    Q_value[index[id_1]]=R1+R2*discount+R3*discount**2+R4*discount**3
                    action_id[id_1] = [id_1, id_2, id_3, id_4]
                    index[id_1]=index[id_1]+1
                    
    Q_value_opt= [{}]*action_space.size
    index_opt = [{}]*action_space.size
    for id in range(0, action_space.size):
        Q_value_opt[id]=max(Q_value[id])
        index_opt[id]=Q_value[id].index(max(Q_value[id]))
    
    id_opt = Q_value_opt.index(max(Q_value_opt))
    
    Action_id = action_id[id_opt]
    return Q_value_opt, Action_id
