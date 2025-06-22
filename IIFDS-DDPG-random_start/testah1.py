import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
"""这个文件是针对单动态障碍的测试环境，测试后打开matlab运行test.m即可得到可视化结果"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from IIFDS import IIFDS
from Method import getReward, transformAction, drawActionCurve

if __name__ == "__main__":
    actionh_matrix=np.loadtxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/data_csv/action_h.csv', delimiter=',')
    actionm_matrix=np.loadtxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/data_csv/action_m.csv', delimiter=',')
    action_matrix=np.loadtxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/data_csv/action.csv', delimiter=',')
    bh_matrix=np.loadtxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/shiyan_csv/b_h4.csv', delimiter=',')
    bm_matrix=np.loadtxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/shiyan_csv/b_m4.csv', delimiter=',')
    actionCurve = np.array([])
    action_stack=[]
    for i in range(119):
      #  bh=bh_matrix[i]
      #  bm=bm_matrix[i]
      #  if bh>bm:
      #     action=actionh_matrix[i]
      #  else:
      #     action=actionm_matrix[i]
      #  action_stack.append(action)
       actionm= actionm_matrix[i]
       actionCurve = np.append(actionCurve, actionm)
    drawActionCurve(actionCurve.reshape(-1,3))
    plt.show()
    #np.savetxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/data_csv/action.csv', action_stack, delimiter=',')