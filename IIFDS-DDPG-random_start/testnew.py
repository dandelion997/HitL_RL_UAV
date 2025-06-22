import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
"""这个文件可以可视化当前实验环境，主要用来保存动作值用来生成人类动作"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
from IIFDS import IIFDS
from Method import getReward, transformAction, drawActionCurve
from config import Config
from pylab import*
mpl.rcParams['font.sans-serif']=['SimHei']

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

font = FontProperties(size=14)

def draw_sphere(ax, pos, r):
    phi, theta = np.mgrid[0.0:2.0*np.pi:60j, 0.0:np.pi:30j]
    x = r*np.sin(theta)*np.cos(phi) + pos[0]
    y = r*np.sin(theta)*np.sin(phi) + pos[1]
    z = r*np.cos(theta) + pos[2]
    return ax.plot_surface(x,y,z, color='b', alpha=0.2)


if __name__ == "__main__":
    num_runs = 10
    tau = 1.0
    I_3 = np.eye(3)
    action_matrix=np.zeros((3, 3))

    conf = Config()
    actionBound = conf.actionBound
    action_stack=[]

    iifds = IIFDS()
    dynamicController = torch.load('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/dynamicActor.pkl',map_location=device)
    actionCurve = np.array([])

    q = np.array(iifds.start)
    qBefore = [None, None, None]
    path = iifds.start.reshape(1,-1)
    rewardSum = 0 

    start = iifds.start
    goal = iifds.goal
    obs_r = iifds.obsR
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制起点和终点
    ax.scatter(start[0], start[1], start[2], s=60, c='cyan', marker='o', edgecolors='k')
    ax.scatter(goal[0], goal[1], goal[2], s=60, c='magenta', marker='o', edgecolors='k')
    ax.text(start[0], start[1], start[2], '  start')
    ax.text(goal[0], goal[1], goal[2], '  goal')


    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_zlabel('z(m)')
    ax.set_title('UAV动态航迹规划路径',fontproperties=font)
    ax.set_box_aspect([1,1,1])

    time_step = 0.1

    for i in range(700):
        action_matrix.fill(0)
        dic = iifds.updateObs(if_test=True)
        vObs, obsCenter, obsCenterNext = dic['v'], dic['obsCenter'], dic['obsCenterNext']
        obs = iifds.calDynamicState(q, obsCenter)
        obs = torch.as_tensor(obs, dtype=torch.float, device=device)
        # obs_Center=np.array(obsCenter)
        # obs_center = obs_Center[:3]
    
        #plt.draw()
        #plt.pause(0.01)
    
        # 模型切换到评估模式
        dynamicController.eval()
        action_sum = np.zeros(conf.act_dim)
        for _ in range(num_runs):
            action = dynamicController(obs).cpu().detach().numpy()
            action_sum += action
            action_matrix+=np.dot(action.T,action)
        action_mean = action_sum / num_runs
        action_mean = transformAction(action_mean, actionBound, conf.act_dim)
        action_stack.append(action_mean)
        print(action_mean)
        Var1=np.dot(np.array(action_mean).T,np.array(action_mean))
        Var2=(1/tau)*I_3+(1/num_runs)* action_matrix
        Var3=Var2-Var1
        Uncertainty=np.max(np.diagonal(Var3).flatten())
        actionCurve = np.append(actionCurve, action)

        # 与环境交互
        qNext = iifds.getqNext(q, obsCenter, vObs, action_mean[0], action_mean[1], action_mean[2], qBefore)
        rewardSum += getReward(obsCenterNext, qNext, q, qBefore, iifds)

        qBefore = q
        q = qNext

        if iifds.distanceCost(q, iifds.goal) < iifds.threshold:
            path = np.vstack((path, iifds.goal))
            _ = iifds.updateObs(if_test=True)
            break
        path = np.vstack((path, q))
        try:
            B1.remove()
        except Exception:
            pass
        
        try:
            B2.remove()
        except Exception:
            pass
        
        B1=draw_sphere(ax,obsCenter,obs_r)
        B2 = ax.scatter(q[0], q[1], q[2], s=80, c='g', marker='^', edgecolors='k')  # 绘制UAV的航路点
        
        if i > 0:
            b1, = ax.plot([obsCenter[0], obsCenterNext[0]],
                          [obsCenter[1], obsCenterNext[1]],
                          [obsCenter[2], obsCenterNext[2]], linewidth=2, color='b')

        # draw_sphere(ax, obs_center, obs_r)  # 绘制动态障碍物的球体
        plt.draw()
        b2, = ax.plot([qBefore[0], q[0]],
                          [qBefore[1], q[1]],
                          [qBefore[2], q[2]], linewidth=2, color='r')
        plt.draw()
        if i == 2:
            ax.legend([b1, b2, B2], ["障碍物移动轨迹", "UAV规划航路", "UAV"], loc='best')

        plt.pause(0.1)
    plt.show()
    np.savetxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/data_csv/pathMatrix.csv', path, delimiter=',')
    np.savetxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/data_csv/action_m.csv', action_stack, delimiter=',')
    #iifds.save_data()
    #routeLen = iifds.calPathLen(path)
    #print('该路径的奖励总和为:%f，路径的长度为:%f' % (rewardSum,routeLen))



