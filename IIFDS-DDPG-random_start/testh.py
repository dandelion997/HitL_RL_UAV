import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
"""This file reads human actions through the keyboard."""
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
    
    iifds = IIFDS()

    q = np.array(iifds.start)
    qBefore = [None, None, None]
    path = iifds.start.reshape(1,-1)
    rewardSum = 0 
    
    
    start = iifds.start
    goal = iifds.goal
    obs_r = iifds.obsR
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the start and goal points
    ax.scatter(start[0], start[1], start[2], s=60, c='cyan', marker='o', edgecolors='k')
    ax.scatter(goal[0], goal[1], goal[2], s=60, c='magenta', marker='o', edgecolors='k')
    ax.text(start[0], start[1], start[2], '  start')
    ax.text(goal[0], goal[1], goal[2], '  goal')


    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_zlabel('z(m)')
    ax.set_title('UAV Dynamic Trajectory Planning Path', fontproperties=font)
    ax.set_box_aspect([1, 1, 1])

    time_step = 0.1

    for i in range(700):
        dic = iifds.updateObs(if_test=True)
        vObs, obsCenter, obsCenterNext = dic['v'], dic['obsCenter'], dic['obsCenterNext']
        obs = iifds.calDynamicState(q, obsCenter)
        obs = torch.as_tensor(obs, dtype=torch.float, device=device)
        # obs_Center=np.array(obsCenter)
        # obs_center = obs_Center[:3]
    
        #plt.draw()
        #plt.pause(0.01)

        # Read human actions through the keyboard
        a1=input("Please enter the first action:")
        a1=float(a1)
        a2=input("Please enter the second action:")
        a2=float(a2)
        a3=input("Please enter the third action:")
        a3=float(a3)
        action=[a1,a2,a3]

        # Interact with the environment
        qNext = iifds.getqNext(q, obsCenter, vObs, action[0], action[1], action[2], qBefore)
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

        # draw_sphere(ax, obs_center, obs_r)  # Draw the dynamic obstacle sphere
        plt.draw()
        b2, = ax.plot([qBefore[0], q[0]],
                          [qBefore[1], q[1]],
                          [qBefore[2], q[2]], linewidth=2, color='r')
        plt.draw()
        if i == 2:
            ax.legend([b1, b2, B2], ["Obstacle Moving Trajectory", "UAV Planned Path", "UAV"], loc='best')

        plt.pause(0.1)
    plt.show()
    np.savetxt('/home/prolee/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/data_csv/pathMatrix.csv', path, delimiter=',')





