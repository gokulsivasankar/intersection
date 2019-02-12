import numpy as np
import scipy.linalg

def get_params():
    class Bunch:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)
    
    params = Bunch(
                w_lane = 4,  
                l_car = 5 ,  
                w_car = 2 ,  
                v_nominal = 2.5, 
                v_max = 5, 
                v_min = 0, 
                t_step_DT = 0.5,   
                t_step_Sim = 0.25, 
                discount = 0.8, 
                dR_drop = -2000, 
                num_cars = 2, 
                max_episode = 1, 
                outfile = 'Test.gif')

    params.complete_flag = np.zeros((params.max_episode,params.num_cars))
    return params