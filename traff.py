import numpy as np

def initial():
    class Bunch:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)
    traffic = Bunch( 
        x = np.empty(shape=[1, 0]),     
        y = np.empty(shape=[1, 0]), 
        orientation = np.empty(shape=[1, 0]),
        v_car = np.empty(shape=[1, 0]), 
        target_car = np.empty(shape=[1, 0])
        )
    return traffic

def update(traffic, x_car, y_car, orientation_car, v_car, target_car):
    traffic.x = np.append(traffic.x, [[x_car]], axis=1 )
    traffic.y = np.append(traffic.y, [[y_car]], axis=1)
    traffic.orientation = np.append(traffic.orientation, [[orientation_car]], axis=1)
    traffic.v_car = np.append(traffic.v_car, [[v_car]], axis=1)
    traffic.target_car = np.append(traffic.target_car, [[target_car]], axis=1)
    return traffic
