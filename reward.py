import math
import numpy as np
from shapely.geometry import Polygon


def reward(X_old, car_id, action_id, params):

    episode = params.episode
    complete_flag = params.complete_flag

    l_car = params.l_car
    w_car = params.w_car
    w_lane = params.w_lane

    # Off road penalty
    Off_road = 0
    Off_road_Penalty = -5000

    l_car_safe = 1.2*l_car     # 1.2
    w_car_safe = 1.2*w_car
    Ego_rectangle = Polygon(
        [[X_old[1,car_id]-l_car_safe/2*math.cos(X_old[3,car_id])+w_car_safe/2*math.sin(X_old[3,car_id]),X_old[2,car_id]-l_car_safe/2*math.sin(X_old[3,car_id])-w_car_safe/2*math.cos(X_old[3,car_id])],
        [X_old[1,car_id]-l_car_safe/2*math.cos(X_old[3,car_id])-w_car_safe/2*math.sin(X_old[3,car_id]), X_old[2,car_id]-l_car_safe/2*math.sin(X_old[3,car_id])+w_car_safe/2*math.cos(X_old[3,car_id])],
        [X_old[1,car_id]+l_car_safe/2*math.cos(X_old[3,car_id])-w_car_safe/2*math.sin(X_old[3,car_id]), X_old[2,car_id]+l_car_safe/2*math.sin(X_old[3,car_id])+w_car_safe/2*math.cos(X_old[3,car_id])],
        [X_old[1,car_id]+l_car_safe/2*math.cos(X_old[3,car_id])+w_car_safe/2*math.sin(X_old[3,car_id]), X_old[2,car_id]+l_car_safe/2*math.sin(X_old[3,car_id])-w_car_safe/2*math.cos(X_old[3,car_id])]])

    RoadBound_rectangle_RD = Polygon(
        [[w_lane,-6*w_lane],
        [w_lane,-2*w_lane],
        [2*w_lane,-w_lane],
        [6*w_lane,-w_lane],
        [6*w_lane,-6*w_lane]])

    RoadBound_rectangle_RU = Polygon(
        [[w_lane,2*w_lane],
        [w_lane,6*w_lane],
        [6*w_lane,6*w_lane],
        [6*w_lane,w_lane],
        [2*w_lane,w_lane]])

    RoadBound_rectangle_LU = Polygon(
        [[-w_lane,6*w_lane],
        [-w_lane,2*w_lane],
        [-2*w_lane,w_lane],
        [-6*w_lane,w_lane],
        [-6*w_lane,6*w_lane]])

    RoadBound_rectangle_LD = Polygon(
        [[-6*w_lane,-w_lane],
        [-2*w_lane,-w_lane],
        [-w_lane,-2*w_lane],
        [-w_lane,-6*w_lane],
        [-6*w_lane,-6*w_lane]])

    if (Ego_rectangle.intersects(RoadBound_rectangle_RD) or 
        Ego_rectangle.intersects(RoadBound_rectangle_RU) or 
        Ego_rectangle.intersects(RoadBound_rectangle_LU) or 
        Ego_rectangle.intersects(RoadBound_rectangle_LD)):
            Off_road = Off_road + Off_road_Penalty

    # Cross mid penalty
    Off_Mid_Penalty = -500

    para = 2
    err = 0

    RoadMid_rectangle_BR = Polygon(
        [[w_lane,-para*w_lane],
        [w_lane,-6*w_lane],
        [-err,-6*w_lane],
        [-err,-para*w_lane]])

    RoadMid_rectangle_BL = Polygon(
        [[err,-para*w_lane],
        [err,-6*w_lane],
        [-w_lane,-6*w_lane],
        [-w_lane,-para*w_lane]])

    RoadMid_rectangle_RU = Polygon(
        [[6*w_lane,-err],
        [para*w_lane,-err],
        [para*w_lane,w_lane],
        [6*w_lane,w_lane]])

    RoadMid_rectangle_RD = Polygon(
        [[6*w_lane,err],
        [6*w_lane,-w_lane],
        [para*w_lane,-w_lane],
        [para*w_lane,err]])

    RoadMid_rectangle_UR = Polygon(
        [[w_lane,para*w_lane],
        [-err,para*w_lane],
        [-err,6*w_lane],
        [w_lane,6*w_lane]])

    RoadMid_rectangle_UL = Polygon(
        [[err,para*w_lane],
        [-w_lane,para*w_lane],
        [-w_lane,6*w_lane],
        [err,6*w_lane]])

    RoadMid_rectangle_LU = Polygon(
        [[-para*w_lane,-err],
        [-6*w_lane,-err],
        [-6*w_lane,w_lane],
        [-para*w_lane,w_lane]])

    RoadMid_rectangle_LD = Polygon(
        [[-para*w_lane,err],
        [-para*w_lane,-w_lane],
        [-6*w_lane,-w_lane],
        [-6*w_lane,err]])

    Direction_vector = [[math.cos(X_old(3,car_id))],[math.sin(X_old(3,car_id))]]
    if np.dot(Direction_vector,[0,1])<0:
        if Ego_rectangle.intersects(RoadMid_rectangle_BR):
            Off_road = Off_road + Off_Mid_Penalty

    if np.dot(Direction_vector,[0,-1])<0:
        if Ego_rectangle.intersects(RoadMid_rectangle_BL):
            Off_road = Off_road + Off_Mid_Penalty

    if np.dot(Direction_vector,[-1,0])<0:
        if Ego_rectangle.intersects(RoadMid_rectangle_RU):
            Off_road = Off_road + Off_Mid_Penalty

    if np.dot(Direction_vector,[1,0])<0:
        if Ego_rectangle.intersects(RoadMid_rectangle_RD):
            Off_road = Off_road + Off_Mid_Penalty

    if np.dot(Direction_vector,[0,1])<0:
        if Ego_rectangle.intersects(RoadMid_rectangle_UR):
            Off_road = Off_road + Off_Mid_Penalty

    if np.dot(Direction_vector,[0,-1])<0:
        if Ego_rectangle.intersects(RoadMid_rectangle_UL):
            Off_road = Off_road + Off_Mid_Penalty

    if np.dot(Direction_vector,[-1,0])<0:
        if Ego_rectangle.intersects(RoadMid_rectangle_LU):
            Off_road = Off_road + Off_Mid_Penalty

    if np.dot(Direction_vector,[1,0])<0:
        if Ego_rectangle.intersects(RoadMid_rectangle_LD):
            Off_road = Off_road + Off_Mid_Penalty


    # Collision penalty
    Colli = 0
    Colli_Penalty = -10000
    l_car_safe = 1.1*l_car     # 1.1
    w_car_safe = 1.1*w_car

    for id in range(1,len(X_old[1,:])):
        if id!=car_id:
            Other_rectangle = Polygon(
                [[X_old[1,id]-l_car_safe/2*math.cos(X_old[3,id])+w_car_safe/2*math.sin(X_old[3,id]), X_old[2,id]-l_car_safe/2*math.sin(X_old[3,id])-w_car_safe/2*math.cos(X_old[3,id])],
                [X_old[1,id]-l_car_safe/2*math.cos(X_old[3,id])-w_car_safe/2*math.sin(X_old[3,id]), X_old[2,id]-l_car_safe/2*math.sin(X_old[3,id])+w_car_safe/2*math.cos(X_old[3,id])],
                [X_old[1,id]+l_car_safe/2*math.cos(X_old[3,id])-w_car_safe/2*math.sin(X_old[3,id]), X_old[2,id]+l_car_safe/2*math.sin(X_old[3,id])+w_car_safe/2*math.cos(X_old[3,id])],
                [X_old[1,id]+l_car_safe/2*math.cos(X_old[3,id])+w_car_safe/2*math.sin(X_old[3,id]), X_old[2,id]+l_car_safe/2*math.sin(X_old[3,id])-w_car_safe/2*math.cos(X_old[3,id])]])

            if Ego_rectangle.intersects(Other_rectangle):
                Colli = Colli + Colli_Penalty


    # Safe zone violation penalty
    Safe = 0
    Safe_Penalty = -1000        # 500

    l_car_safe_front = 2*l_car
    l_car_safe_back = 1.2*l_car
    w_car_safe = 1.2*w_car

    Ego_rectangle = Polygon(
        [[X_old[1,car_id]-l_car_safe_back/2*math.cos(X_old[3,car_id])+w_car_safe/2*math.sin(X_old[3,car_id]),X_old[2,car_id]-l_car_safe_back/2*math.sin(X_old[3,car_id])-w_car_safe/2*math.cos(X_old[3,car_id])],
        [X_old[1,car_id]-l_car_safe_back/2*math.cos(X_old[3,car_id])-w_car_safe/2*math.sin(X_old[3,car_id]), X_old[2,car_id]-l_car_safe_back/2*math.sin(X_old[3,car_id])+w_car_safe/2*math.cos(X_old[3,car_id])],
        [X_old[1,car_id]+l_car_safe_front/2*math.cos(X_old[3,car_id])-w_car_safe/2*math.sin(X_old[3,car_id]), X_old[2,car_id]+l_car_safe_front/2*math.sin(X_old[3,car_id])+w_car_safe/2*math.cos(X_old[3,car_id])],
        [X_old[1,car_id]+l_car_safe_front/2*math.cos(X_old[3,car_id])+w_car_safe/2*math.sin(X_old[3,car_id]), X_old[2,car_id]+l_car_safe_front/2*math.sin(X_old[3,car_id])-w_car_safe/2*math.cos(X_old[3,car_id])]])
    
    for id in range(1,len(X_old[1,:])):
        if id!=car_id:
            Other_rectangle = Polygon(
                [[X_old[1,id]-l_car_safe_back/2*math.cos(X_old[3,id])+w_car_safe/2*math.sin(X_old[3,id]), X_old[2,id]-l_car_safe_back/2*math.sin(X_old[3,id])-w_car_safe/2*math.cos(X_old[3,id])],
                [X_old[1,id]-l_car_safe_back/2*math.cos(X_old[3,id])-w_car_safe/2*math.sin(X_old[3,id]), X_old[2,id]-l_car_safe_back/2*math.sin(X_old[3,id])+w_car_safe/2*math.cos(X_old[3,id])],
                [X_old[1,id]+l_car_safe_front/2*math.cos(X_old[3,id])-w_car_safe/2*math.sin(X_old[3,id]), X_old[2,id]+l_car_safe_front/2*math.sin(X_old[3,id])+w_car_safe/2*math.cos(X_old[3,id])],
                [X_old[1,id]+l_car_safe_front/2*math.cos(X_old[3,id])+w_car_safe/2*math.sin(X_old[3,id]), X_old[2,id]+l_car_safe_front/2*math.sin(X_old[3,id])-w_car_safe/2*math.cos(X_old[3,id])]])

            if Ego_rectangle.intersects(Other_rectangle):
                Safe = Safe + Safe_Penalty

    # # Safe zone violation penalty
    # Safe = 0
    # Safe_Penalty = -2000
    # for id in range(1,len(X_old[1,:])):
    #    if id!=car_id:
    #     Distance_2 = (X_old[1,car_id]-X_old[1,id])^2+(X_old[2,car_id]-X_old[2,    id])^2
    #     if(Distance_2<1*(l_car^2+w_car^2)):    
    # #         Safe = Safe+Safe_Penalty
    #           Safe = Safe+Safe_Penalty*(l_car^2+w_car^2)/Distance_2


    # Completion reward
    Complete = 0
    Complete_Penalty = -50

    if X_old[5,car_id] == 1:
        if X_old[1,car_id]<6*w_lane:
            Complete = Complete + 1*Complete_Penalty*abs(6*w_lane-X_old[1,car_id])
            Complete = Complete + 2*Complete_Penalty*abs(-w_lane/2-(X_old[2,car_id]+0.1*math.sin(X_old[3,car_id])))
        else:
            complete_flag[episode,car_id] = 1
 
    elif X_old[5,car_id] == 2:
        if (X_old[2,car_id]<6*w_lane):
            Complete = Complete + 1*Complete_Penalty*abs(6*w_lane-X_old[2,car_id])
            Complete = Complete + 2*Complete_Penalty*abs(w_lane/2-(X_old[1,car_id]+0.1*math.cos(X_old[3,car_id])))    
        else:
            complete_flag[episode,car_id] = 1

    elif X_old[5,car_id] == 3:
        if (X_old[1,car_id]>-6*w_lane):
            Complete = Complete + 1*Complete_Penalty*abs(-6*w_lane-X_old[1,car_id])
            Complete = Complete + 2*Complete_Penalty*abs(w_lane/2-(X_old[2,car_id]+0.1*math.sin(X_old[3,car_id])))    
        else:
            complete_flag[episode,car_id] = 1
    
    elif X_old[5,car_id] == 4:
        if (X_old[2,car_id]>-6*w_lane):
            Complete = Complete + 1*Complete_Penalty*abs(-6*w_lane-X_old[2,car_id])
            Complete = Complete + 2*Complete_Penalty*abs(-w_lane/2-(X_old[1,car_id]+0.1*math.cos(X_old[3,car_id])))    
        else:
            complete_flag[episode,car_id] = 1


    # Speed reward
    Speed = -5*1/(X_old[4,car_id]+0.1)

    # Effort penalty
    if(action_id!=1):
        Effort = 0
    else:
        Effort = 0

    # Specific penalty
    Specific_Penalty = -300
    if(car_id==1):
        SpecificBound = Polygon(
            [[err,6*w_lane],
            [err,-6*w_lane],
            [-6*w_lane,-6*w_lane],
            [-6*w_lane,6*w_lane]])

        if Ego_rectangle.intersects(SpecificBound):
                Off_road = Off_road + Specific_Penalty

    elif(car_id==2):
        SpecificBound = Polygon(
            [[-2*w_lane,w_lane],
            [w_lane,-2*w_lane],
            [w_lane,-6*w_lane],
            [-6*w_lane,-6*w_lane],
            [-6*w_lane,w_lane]])

        if Ego_rectangle.intersects(SpecificBound):
                Off_road = Off_road + Specific_Penalty

    R = (Off_road + Colli + Safe + Complete + Speed + Effort)
    params.complete_flag = complete_flag

    return R, params