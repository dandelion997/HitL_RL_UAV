import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
"""这个文件是针对单动态障碍的测试环境，测试后打开matlab运行test.m即可得到可视化结果"""

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
    dynamicController = torch.load('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-SAC-random_start/TrainedModel/dynamicActor.pkl',map_location=device)
    actionCurve = np.array([])

    q = iifds.start
    qBefore = [None, None, None]
    path = iifds.start.reshape(1,-1)
    reward_stack=[]
    rewardSum = 0
    for i in range(500):
        dic = iifds.updateObs(if_test=True)
        vObs, obsCenter, obsCenterNext = dic['v'], dic['obsCenter'], dic['obsCenterNext']
        obs = iifds.calDynamicState(q, obsCenter)
        obs = torch.as_tensor(obs, dtype=torch.float, device=device)
        action = dynamicController.act(obs, deterministic=True)
        action = transformAction(action, actionBound, conf.act_dim)
        actionCurve = np.append(actionCurve, action)
        # 与环境交互
        qNext = iifds.getqNext(q, obsCenter, vObs, action[0], action[1], action[2], qBefore)
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

    drawActionCurve(actionCurve.reshape(-1,3))
    # np.savetxt('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/env2_csv/pathMatrix_sac2.csv', path, delimiter=',')
    # np.savetxt('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/env2_csv/reward_sac2.csv', reward_stack, delimiter=',')
    # iifds.save_data()
    routeLen = iifds.calPathLen(path)
    print('该路径的奖励总和为:%f，路径的长度为:%f' % (rewardSum,routeLen))
    plt.show()

