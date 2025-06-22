import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
"""这个文件是用人类动作得到运动轨迹，针对第四个实验场景，运行test.m即可得到可视化结果"""
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
    action_matrix=np.loadtxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/data_csv/action_h1.csv', delimiter=',')
    actionCurve = np.array([])

    q = iifds.start
    qBefore = [None, None, None]
    path = iifds.start.reshape(1,-1)
    action_stack=[]
    reward_stack=[]
    rewardSum = 0
    for i in range(500):
        dic = iifds.updateObs(if_test=True)
        vObs, obsCenter, obsCenterNext = dic['v'], dic['obsCenter'], dic['obsCenterNext']
        obs = iifds.calDynamicState(q, obsCenter)
        obs = torch.as_tensor(obs, dtype=torch.float, device=device)
        action_m = action_matrix[i]
        action_stack.append(action_m)
        actionCurve = np.append(actionCurve, action_m)
        # 与环境交互
        qNext = iifds.getqNext(q, obsCenter, vObs, action_m[0], action_m[1], action_m[2], qBefore)
        r=getReward(obsCenterNext, qNext, q, qBefore, iifds)
        print(r)
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
    np.savetxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/data_csv/pathMatrix.csv', path, delimiter=',')
    #np.savetxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/shiyan_csv/pathMatrix_h4.csv', path, delimiter=',')
    #np.savetxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/shiyan_csv/reward_h4.csv', reward_stack, delimiter=',')
    iifds.save_data()
    routeLen = iifds.calPathLen(path)
    print('该路径的奖励总和为:%f，路径的长度为:%f' % (rewardSum,routeLen))
    plt.show()
