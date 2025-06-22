import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
"""这个文件是针对单动态障碍的测试环境，测试后打开matlab运行test.m即可得到可视化结果"""
import torch
import math
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from IIFDS import IIFDS
from Method import getReward, transformAction, drawActionCurve
from config import Config


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def integrand3(x,u):
    return (u/x)*0.85**x*np.log(1/0.85)

def integrand4(x):
    return 0.85**x*np.log(1/0.85)

if __name__ == "__main__":
    num_runs = 10
    tau = 1.0
    I_3 = np.eye(3)
    action_matrix=np.zeros((3, 3)) 
    
    conf = Config()
    actionBound = conf.actionBound

    iifds = IIFDS()
    dynamicController = torch.load('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/dynamicActor.pkl',map_location=device)
    actionh_matrix=np.loadtxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/data_csv/action_h.csv', delimiter=',')
    bh_matrix=np.loadtxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/shiyan_csv/b_h4.csv', delimiter=',')
    actionCurve = np.array([])

    q = iifds.start
    qBefore = [None, None, None]
    path = iifds.start.reshape(1,-1)
    #action_stack=[]
    rewardSum = 0
    for i in range(500):
        action_matrix.fill(0)
        action_sum=0
        dic = iifds.updateObs(if_test=True)
        vObs, obsCenter, obsCenterNext = dic['v'], dic['obsCenter'], dic['obsCenterNext']
        obs = iifds.calDynamicState(q, obsCenter)
        obs = torch.as_tensor(obs, dtype=torch.float, device=device)
        # 模型切换到评估模式
        dynamicController.eval()
        action_sum = np.zeros(conf.act_dim)
        for _ in range(num_runs):
            action = dynamicController(obs).cpu().detach().numpy()
            action_sum += action
            action_matrix+=np.dot(action.T,action)
        action_m = action_sum / num_runs
        action_m = transformAction(action_m, actionBound, conf.act_dim)
        action_h=actionh_matrix[i]
        #action_stack.append(action_mean)
        Var1=np.dot(np.array(action_m).T,np.array(action_m))
        Var2=(1/tau)*I_3+(1/num_runs)* action_matrix
        Var3=Var2-Var1
        Uncertainty=abs(np.max(np.diagonal(Var3).flatten()))
        fenzi1,error1=integrate.quad(integrand3,Uncertainty,np.inf,args=(Uncertainty,))
        fenmu1,error2=integrate.quad(integrand4,Uncertainty,np.inf)
        b_m=1.5-fenzi1/fenmu1
        b_h=bh_matrix[i]
        action_mh=(b_m/(b_m+b_h))*np.array(action_m)+(b_h/(b_m+b_h))*np.array(action_h)
        actionCurve = np.append(actionCurve, action_mh)
        
        # 与环境交互
        qNext = iifds.getqNext(q, obsCenter, vObs, action_mh[0], action_mh[1], action_mh[2], qBefore)
        rewardSum += getReward(obsCenterNext, qNext, q, qBefore, iifds)

        qBefore = q
        q = qNext

        if iifds.distanceCost(q, iifds.goal) < iifds.threshold:
            path = np.vstack((path, iifds.goal))
            _ = iifds.updateObs(if_test=True)
            break
        path = np.vstack((path, q))
        
        

    drawActionCurve(actionCurve.reshape(-1,3))
    np.savetxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/data_csv/pathMatrix.csv', path, delimiter=',')
    #np.savetxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/data_csv/action_m.csv', action_stack, delimiter=',')
    
    iifds.save_data()
    routeLen = iifds.calPathLen(path)
    print('该路径的奖励总和为:%f，路径的长度为:%f' % (rewardSum,routeLen))
    plt.show()


