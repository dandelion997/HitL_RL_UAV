# File func: various methods

import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from Multi_obstacle_environment_test import Environment
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getReward(obsCenter, qNext, q, qBefore, iifds):
    """
    Obtain the reward value function in reinforcement learning
    """
    distance = iifds.distanceCost(qNext, obsCenter)
    flag = True if distance <= iifds.obsR else False
    reward = 0
    if flag: # Intersecting with the obstacle.
       reward += (distance - iifds.obsR)/iifds.obsR - 1
    else:
        if distance < iifds.obsR + 0.4:   # Threat zone
            tempR = iifds.obsR + 0.4
            reward += (distance-tempR)/tempR-0.3
        distance1 = iifds.distanceCost(qNext, iifds.goal)
        distance2 = iifds.distanceCost(iifds.start, iifds.goal)
        if distance1 > iifds.threshold:
            reward += -distance1/distance2
        else:
            reward += -distance1/distance2 + 3
    """Acquiring the Reward Value Function: Version 2"""
    # distance = iifds.distanceCost(qNext, obsCenter)
    # flag = True if distance <= iifds.obsR else False
    # reward = 0
    # r_len=0
    # r_col=0
    # if flag: # Intersecting with the obstacle.
    #     r_col = (distance - iifds.obsR)/iifds.obsR - 1
    # else:
    #     r_col = (distance - iifds.obsR)/iifds.obsR
    # distance1 = iifds.distanceCost(qNext, iifds.goal)
    # distance2 = iifds.distanceCost(iifds.start, iifds.goal)
    # if distance1 > iifds.threshold:
    #     r_len = -distance1/distance2
    # else:
    #     r_len = -distance1/distance2 + 3 
    # reward=0.5*r_col+0.5*r_len 

    return reward

def get_reward_multiple(env,qNext,dic):
    """Obtain the reward function for multiple dynamic obstacles"""
    reward = 0
    distance = env.distanceCost(qNext,dic['obsCenter'])
    if distance<=dic['obs_r']:
        reward += (distance-dic['obs_r'])/dic['obs_r']-1
    else:
        if distance < dic['obs_r'] + 0.4:
            tempR = dic['obs_r'] + 0.4
            reward += (distance-tempR)/tempR-0.3
        distance1 = env.distanceCost(qNext, env.goal)
        distance2 = env.distanceCost(env.start, env.goal)
        if distance1 > env.threshold:
            reward += -distance1 / distance2
        else:
            reward += -distance1 / distance2 + 5
    return reward


def drawActionCurve(actionCurveList):
    """
    :param actionCurveList: List of action-values
    :return: None Plotting the image
    """
    plt.figure()
    for i in range(actionCurveList.shape[1]):
        array = actionCurveList[:, i]
        if i == 0: label = 'row0'
        if i == 1: label = 'sigma0'
        if i == 2: label = 'theta'
        plt.plot(np.arange(array.shape[0]), array, linewidth=2, label=label)
    plt.title('Variation diagram')
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend(loc='best')

def checkPath(apf):
    sum = 0
    for i in range(apf.path.shape[0] - 1):
        sum += apf.distanceCost(apf.path[i,:], apf.path[i+1,:])
    for i, j in zip(apf.path, apf.dynamicSphere_Path):
        if apf.distanceCost(i,j) <= apf.dynamicSphere_R:
            print('There is an intersection with the obstacle, and the trajectory distance is:', sum)
            return
    print('No intersection with the obstacle, trajectory distance is:', sum)

def transformAction(actionBefore, actionBound, actionDim):
    """Map the actions output by reinforcement learning to the specified action range"""
    actionAfter = []
    for i in range(actionDim):
        action_i = actionBefore[i]
        action_bound_i = actionBound[i]
        actionAfter.append((action_i+1)/2*(action_bound_i[1] - action_bound_i[0]) + action_bound_i[0])
    return actionAfter


def test(iifds, pi, conf):
    """Test the training effect in a dynamic single-obstacle environment"""
    iifds.reset()    # Reset the environment
    q = iifds.start
    qBefore = [None, None, None]
    rewardSum = 0
    for i in range(500):
        dic = iifds.updateObs(if_test=True)
        vObs, obsCenter, obsCenterNext = dic['v'], dic['obsCenter'], dic['obsCenterNext']
        obs = iifds.calDynamicState(q, obsCenter)
        obs = torch.as_tensor(obs, dtype=torch.float, device=device)
        action = pi(obs).cpu().detach().numpy()
        action = transformAction(action, conf.actionBound, conf.act_dim)
        # Interact with the environment
        qNext = iifds.getqNext(q, obsCenter, vObs, action[0], action[1], action[2], qBefore)
        rewardSum += getReward(obsCenterNext, qNext, q, qBefore, iifds)

        qBefore = q
        q = qNext

        if iifds.distanceCost(q, iifds.goal) < iifds.threshold:
            break
    return rewardSum

def test_multiple(pi, conf):
    """Test the model performance in a dynamic multi-obstacle environment"""
    reward_list = []
    for index in range(1,7):      # There are 6 test environments from 1 to 6
        env = Environment(index)
        env.reset()
        q = env.start
        qBefore = [None, None, None]
        rewardSum = 0
        for i in range(500):
            data_dic = env.update_obs_pos(q)
            v_obs, obs_center, obs_R = data_dic['v'], data_dic['obsCenter'], data_dic['obs_r']
            state = env.calDynamicState(q, obs_center, obs_R, v_obs)
            state = torch.as_tensor(state, dtype=torch.float, device=device)
            action = pi(state).cpu().detach().numpy()
            a = transformAction(action, conf.actionBound, conf.act_dim)
            qNext = env.getqNext(q, obs_center, v_obs, obs_R, a[0], a[1], a[2], qBefore)
            rewardSum += get_reward_multiple(env,qNext,data_dic)
            qBefore = q
            q = qNext
            if env.distanceCost(q, env.goal) < env.threshold:
                break
        reward_list.append(rewardSum)
    return reward_list


def setup_seed(seed):
    """Function to set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

