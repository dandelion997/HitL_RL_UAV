import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from IIFDS import IIFDS
from Method import getReward, transformAction, drawActionCurve
from config import Config


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    conf = Config()
    actionBound = conf.actionBound

    iifds = IIFDS()
    dynamicController = torch.load('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/dynamicActor.pkl',map_location=device)
    q_network = torch.load('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/dynamicCritic.pkl',map_location=device)
    humanController = torch.load('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/dynamicActor_h.pkl',map_location=device)
    actionCurve = np.array([])

    q = iifds.start
    qBefore = [None, None, None]
    path = iifds.start.reshape(1,-1)
    action_stack=[]
    reward_stack=[]
    theta_stack=[]
    omega_stack=[]
    rewardSum = 0
    # Initial values and related parameters
    R=3
    gamma=0.99
    a_length=3
    K_1=1/4
    K_2=(np.sqrt(3)-1)*(2-np.sqrt(3))/((3-np.sqrt(3))**3)
    theta= np.array([0.0,0.0,0.0])
    #theta= np.array([1.0,1.0,1.0])
    omega=1/(1+np.exp(-theta))
    #sigma=np.array([1.0,1.0,1.0])
    sigma=np.array([np.sqrt(0.2),np.sqrt(0.2),np.sqrt(0.2)])
    covariance=np.diag(sigma)
    alpha=0.1
    for i in range(500):
        dic = iifds.updateObs(if_test=True)
        vObs, obsCenter, obsCenterNext = dic['v'], dic['obsCenter'], dic['obsCenterNext']
        obs = iifds.calDynamicState(q, obsCenter)
        obs = torch.as_tensor(obs, dtype=torch.float, device=device)
        action_m = dynamicController(obs).cpu().detach().numpy()
        action_m = transformAction(action_m, actionBound, conf.act_dim)
        #print(action_m)
        action_h = humanController(obs).cpu().detach().numpy()
        action_h = transformAction(action_h, actionBound, conf.act_dim)
        #print(action_h)
        #action_stack.append(action)

        # Compute the Q-value.
        action_tensor = torch.as_tensor(action_m, dtype=torch.float, device=device)
        q_value=q_network(obs,action_tensor)
        q_value = q_value.item()
        #print(q_value)

        # Compute the value of the approximate gradient.
        deta_J=((np.array(action_m)-np.array(action_h))*omega*(1-omega)*q_value)/(sigma**2)
        #print(deta_J)

        # Compute the weight parameters.
        theta=theta+alpha*deta_J
        #print(theta)
        theta_stack.append(theta)
        omega=1/(1+np.exp(-theta))
        #print(omega)
        omega_stack.append(omega)
        # Human-machine hybrid action
        # mu=omega*np.array(action_m)+(1-omega)*np.array(action_h)
        # action=np.random.multivariate_normal(mu,covariance)
        action=omega*np.array(action_m)+(1-omega)*np.array(action_h)
        # #print(action)
        actionCurve = np.append(actionCurve, action)
        # Interact with the environment
        qNext = iifds.getqNext(q, obsCenter, vObs, action[0], action[1], action[2], qBefore)
        r=getReward(obsCenterNext, qNext, q, qBefore, iifds)
        reward_stack.append(r)
        rewardSum += r

        qBefore = q
        q = qNext

        if iifds.distanceCost(q, iifds.goal) < iifds.threshold:
            path = np.vstack((path, iifds.goal))
            _ = iifds.updateObs(if_test=True)
            break
        path = np.vstack((path, q))

    drawActionCurve(actionCurve.reshape(-1,3))
    
    routeLen = iifds.calPathLen(path)
    print('The total reward for this path is: %f, and the length of the path is: %f' % (rewardSum, routeLen))
    plt.show()
