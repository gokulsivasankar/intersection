import math

def motion_update(X_old, car_id, action_id, t_step, params):

    # 1:maintian  2:turn left  3:turn right  4:accelerate  5:decelerate  6:hard brake    
    
    if action_id == '1':
      X_old[4,car_id] = X_old[4,car_id]
      X_old[3,car_id] = X_old[3,car_id]
      X_old[1,car_id] = (X_old[1,car_id]+X_old[4,car_id]*math.cos(X_old[3,car_id])*t_step)
      X_old[2,car_id] = (X_old[2,car_id]+X_old[4,car_id]*math.sin(X_old[3,car_id])*t_step)

    elif action_id == '2':
        X_old[4,car_id] = X_old[4,car_id]
        X_old[3,car_id] = X_old[3,car_id]+math.pi/4*t_step
        X_old[1,car_id] = (X_old[1,car_id]+X_old[4,car_id]*math.cos(X_old[3,car_id])*t_step)
        X_old[2,car_id] = (X_old[2,car_id]+X_old[4,car_id]*math.sin(X_old[3,car_id])*t_step)

    elif action_id == '3':
        X_old[4,car_id] = X_old[4,car_id]
        X_old[3,car_id] = X_old[3,car_id]-math.pi/4*t_step
        X_old[1,car_id] = (X_old[1,car_id]+X_old[4,car_id]*math.cos(X_old[3,car_id])*t_step)
        X_old[2,car_id] = ([X_old[2,car_id]+X_old[4,car_id]*math.sin(X_old[3,car_id])*t_step])

    elif action_id == '4':
        X_old[4,car_id] = X_old[4,car_id]+2.5*t_step
        if X_old[4,car_id]>params.v_max:
            X_old[4,car_id]=params.v_max
        X_old[3,car_id] = X_old[3,car_id]
        X_old[1,car_id] = (X_old[1,car_id]+X_old[4,car_id]*math.cos(X_old[3,car_id])*t_step)
        X_old[2,car_id] = (X_old[2,car_id]+X_old[4,car_id]*math.sin(X_old[3,car_id])*t_step)
    
    elif action_id == '5':
        X_old[4,car_id] = X_old[4,car_id]-2.5*t_step
        if X_old[4,car_id]<params.v_min:
            X_old[4,car_id]=params.v_min
        X_old[3,car_id] = X_old[3,car_id]
        X_old[1,car_id] = (X_old[1,car_id]+X_old[4,car_id]*math.cos(X_old[3,car_id])*t_step)
        X_old[2,car_id] = (X_old[2,car_id]+X_old[4,car_id]*math.sin(X_old[3,car_id])*t_step)
    
    elif action_id == '6':
        X_old[4,car_id] = X_old[4,car_id]-5*t_step
        if X_old[4,car_id]<params.v_min:
            X_old[4,car_id]=params.v_min
        X_old[3,car_id] = X_old[3,car_id]
        X_old[1,car_id] = (X_old[1,car_id]+X_old[4,car_id]*math.cos(X_old[3,car_id])*t_step)
        X_old[2,car_id] = (X_old[2,car_id]+X_old[4,car_id]*math.sin(X_old[3,car_id])*t_step)

    return X_old