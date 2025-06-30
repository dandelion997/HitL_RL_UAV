#这个文件是用了dropout来计算不确定性
# File func: test
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
    dynamicController = torch.load('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/dynamicActor.pkl',map_location=device)
    actionCurve = np.array([])

    q = iifds.start
    qBefore = [None, None, None]
    path = iifds.start.reshape(1,-1)
    action_stack=[]
    bm_stack=[]
    #path1=np.array([None, None, None])
    rewardSum = 0
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
            action_sum += action
            action_matrix+=np.dot(action.T,action)
        #print(action_matrix)
        action_mean = action_sum / num_runs
        #print(Var1)
        action_mean = transformAction(action_mean, actionBound, conf.act_dim)
        #print(action_mean)
        action_stack.append(action_mean)
        Var1=np.dot(np.array(action_mean).T,np.array(action_mean))
        Var2=(1/tau)*I_3+(1/num_runs)* action_matrix
        Var3=Var2-Var1
        #print(Var3.shape)
        Uncertainty=abs(np.max(np.diagonal(Var3).flatten()))
        fenzi1,error1=integrate.quad(integrand3,Uncertainty,np.inf,args=(Uncertainty,))
        #fenzi=fenzi1+0.2
        fenmu1,error2=integrate.quad(integrand4,Uncertainty,np.inf)
        #fenmu=fenmu1+0.2
        b_m=1.5-fenzi1/fenmu1
        bm_stack.append(b_m)
        
        print(b_m)
        actionCurve = np.append(actionCurve, action_mean)

        # Interact with the environment
        qNext = iifds.getqNext(q, obsCenter, vObs, action_mean[0], action_mean[1], action_mean[2], qBefore)
        rewardSum += getReward(obsCenterNext, qNext, q, qBefore, iifds)

        qBefore = q
        q = qNext

        if iifds.distanceCost(q, iifds.goal) < iifds.threshold:
            path = np.vstack((path, iifds.goal))
            _ = iifds.updateObs(if_test=True)
            break
        path = np.vstack((path, q))
        #path1 = np.vstack((path1, action_mean))
        

    drawActionCurve(actionCurve.reshape(-1,3))
    # np.savetxt('./data_csv/pathMatrix.csv', path, delimiter=',')
    np.savetxt('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/data_csv/pathMatrix.csv', path, delimiter=',')
    np.savetxt('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/data_csv/action_m.csv', action_stack, delimiter=',')
    np.savetxt('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/shiyan_csv/b_m4.csv', bm_stack, delimiter=',')
    iifds.save_data()
    routeLen = iifds.calPathLen(path)
    print('The total reward for this path is: %f, and the length of the path is: %f' % (rewardSum, routeLen))
    plt.show()

