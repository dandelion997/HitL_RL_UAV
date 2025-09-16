import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import math
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from IIFDS import IIFDS
from Method import getReward, transformAction, drawActionCurve
from config import Config


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    num_runs = 10
    tau = 1.0
    I_3 = np.eye(3)
    action_matrix=np.zeros((3, 3)) 
    
    conf = Config()
    actionBound = conf.actionBound

    iifds = IIFDS()
    
    dynamicController = torch.load('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/dynamicActor.pkl',map_location=device)
    humanController = torch.load('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/dynamicActor_h.pkl',map_location=device)
    Q=torch.load('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/dynamicCritic_h.pkl',map_location=device)

    q = iifds.start
    qBefore = [None, None, None]
    path = iifds.start.reshape(1,-1)
    action_stack=[]
    reward_stack=[]
    Q_vec=[]
    #path1=np.array([None, None, None])
    rewardSum = 0
    qvalue=0
    for i in range(500):
        action_matrix.fill(0)
        action_sum=0
        dic = iifds.updateObs(if_test=True)
        vObs, obsCenter, obsCenterNext = dic['v'], dic['obsCenter'], dic['obsCenterNext']
        obs = iifds.calDynamicState(q, obsCenter)
        obs = torch.as_tensor(obs, dtype=torch.float, device=device)
        # Switch the model to evaluation mode.
        dynamicController.eval()
        action_sum = np.zeros(conf.act_dim)
        for _ in range(num_runs):
            action = dynamicController(obs).cpu().detach().numpy()
            qvalue=Q(obs,action).cpu().detach().numpy()
            Q_vec.append(qvalue)
        a_m = action_sum / num_runs
        a_m = transformAction(a_m, actionBound, conf.act_dim)
        a_h=humanController(obs).cpu().detach().numpy()
        a_h=transformAction(a_h, actionBound, conf.act_dim)
        # Action selection
        Q_var=np.var(Q_vec,axis=0)
        Q_diff=abs(Q(obs,a_m).cpu().detach().numpy()-Q(obs,a_h).cpu().detach().numpy())
        if Q_diff >2.4 or Q_var > 0.5:
            a = a_h
        else:
            a = a_m
        qNext = iifds.getqNext(q, obsCenter, vObs, a[0], a[1], a[2], qBefore)
        r= getReward(obsCenterNext, qNext, q, qBefore, iifds)
        reward_stack.append(r)
        rewardSum += r
        qBefore = q
        q = qNext

        if iifds.distanceCost(q, iifds.goal) < iifds.threshold:
            path = np.vstack((path, iifds.goal))
            _ = iifds.updateObs(if_test=True)
            break
        path = np.vstack((path, q))
        
    
    routeLen = iifds.calPathLen(path)
    print('The total reward for this path is: %f, and the length of the path is: %f' % (rewardSum,routeLen))
    plt.show()

