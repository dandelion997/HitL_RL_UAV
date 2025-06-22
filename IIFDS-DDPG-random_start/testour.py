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
    alpha_stack=[]
    rewardSum = 0
    #初值和相关参数的值
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
        
        #计算Q值
        action_tensor = torch.as_tensor(action_m, dtype=torch.float, device=device)
        q_value=q_network(obs,action_tensor)
        q_value = q_value.item()
        #print(q_value)
        
        #计算近似梯度的值
        deta_J=(np.array(action_m)-np.array(action_h))*omega*(1-omega)*q_value/sigma**2
        #print(deta_J)
        #计算最优学习因子
        x=(K_2*abs(np.array(action_h)-np.array(action_m))*a_length)/(np.sqrt(2*np.pi)*(sigma**3)*((1-gamma)**2))
        y=(gamma*(K_1**2)*(abs(np.array(action_h)-np.array(action_m))**2))/(8*(sigma**2)*((1-gamma)**3))
        c_2=-(x+y)*R*np.linalg.norm(deta_J,1)**2
        #print(c_2)
        c_1=np.linalg.norm(deta_J,2)**2
        #print(c_1)
        alpha=-c_1/(2*c_2)
        alpha_stack.append(alpha)
        #print(alpha)
        #计算权重参数
        theta=theta+alpha*deta_J
        #print(theta)
        theta_stack.append(theta)
        omega=1/(1+np.exp(-theta*100000))
        #print(omega)
        omega_stack.append(omega)
        #人机混合动作
        # mu=omega*np.array(action_m)+(1-omega)*np.array(action_h)
        # action=np.random.multivariate_normal(mu,covariance)
        action=omega*np.array(action_m)+(1-omega)*np.array(action_h)
        # #print(action)
        actionCurve = np.append(actionCurve, action)
        # 与环境交互
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
    # np.savetxt('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/env2_csv/pathMatrix_al2.csv', path, delimiter=',')
    # np.savetxt('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/env2_csv/reward_al2.csv', reward_stack, delimiter=',')
    # np.savetxt('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/env2_csv/theta_al2.csv', theta_stack, delimiter=',')
    # np.savetxt('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/env2_csv/omega_our.csv', omega_stack, delimiter=',')
    # np.savetxt('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/env2_csv/alpha.csv', alpha_stack, delimiter=',')
    # #np.savetxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/shiyan_csv/pathMatrix_m4.csv', path, delimiter=',')
    # #np.savetxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/shiyan_csv/reward_m4.csv', reward_stack, delimiter=',')
    # #np.savetxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/data_csv/action_m2.csv', action_stack, delimiter=',')
    # iifds.save_data()
    routeLen = iifds.calPathLen(path)
    print('该路径的奖励总和为:%f，路径的长度为:%f' % (rewardSum,routeLen))
    plt.show()
