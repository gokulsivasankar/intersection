import numpy as np

def initial():
    class Bunch:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)
    
    traffic = Bunch( \
        x = np.empty(shape=[0, 1]), \
        y = np.empty(shape=[0, 1]), \
        v = np.empty(shape=[0, 1]), \
        target = np.empty(shape=[0, 1]),\
        orientation = np.empty(shape=[0, 1]))
    
    return traffic


def update(traffic,x_car,y_car,v_car,orientation_car,target_car):
    traffic.x = np.append(traffic.x, [[x_car]], axis=0)
    traffic.y = np.append(traffic.y, [[y_car]], axis=0)
    traffic.v = np.append(traffic.v, [[v_car]], axis=0)
    traffic.target = np.append(traffic.target, [[target_car]], axis=0)
    traffic.orientraiton = np.append(traffic.orientation, [[orientation_car]],axis=0)

    return traffic