import numpy as np
import scipy.linalg

def get_params():
    class Bunch:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)
    
    # Declare constant parameters
    params = Bunch(
                w_lane = 4,       # (m) lane width
                l_car = 5 ,       # (m) car length
                w_car = 2 ,       # (m) car width
                v_nominal = 2.5,  # (m/s) nominal car speed  
                v_max = 5,        # (m/s) maximum car speed 
                v_min = 0,        # (m/s) minimum car speed
                t_step_DT = 0.5,     # (s) 
                t_step_Sim = 0.25,   # (s)
                discount = 0.8,      # discount factor
                dR_drop = -2000,     # ?
                num_cars = 2,        # number of cars
                max_episode = 1,     # number of maximum episode
                outfile = 'Test.gif')

    params.complete_flag = np.zeros((params.max_episode,params.num_cars))
    return params