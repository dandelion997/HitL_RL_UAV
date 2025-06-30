import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import Config
from dynamic_obstacle_environment import obs_list
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Environment:
    def __init__(self,index):
        """Basic parameters:"""
        self.V0 = 1
        self.threshold = 0.2
        self.step_size = 0.1
        self.lam = 8  # The larger the value, the more the speed of the obstacle is considered

        self.start = np.array([0,2,5],dtype=float)
        self.goal = np.array([10, 10, 5.5], dtype=float)

        self.time_log = 0
        self.time_step = 0.1

        self.xmax = 10 / 180 * np.pi  # Maximum yaw rate, the angle allowed to change at each step
        self.gammax = 10 / 180 * np.pi  # Maximum climb rate, the angle allowed to change at each step
        self.maximumClimbingAngle = 100 / 180 * np.pi  # Maximum climb angle
        self.maximumSubductionAngle = - 75 / 180 * np.pi  # Maximum dive angle

        self.obs_num = 4           # Number of obstacles
        self.obs_r = [1, 1, 1, 1]  # Obstacle radius table

        self.path = {}                 # Dictionary to record the trace of all obstacles
        for i in range(self.obs_num):
            self.path[i] = np.array([[]]).reshape(-1,3)


        if index == 1:                            # Extract obs_num obstacle traces from obs_list
            self.obs = obs_list[0:4]
        elif index == 2:
            self.obs = obs_list[4:8]
        elif index == 3:
            self.obs = [obs_list[i] for i in [0,1,4,5]]
        elif index == 4:
            self.obs = [obs_list[i] for i in [0,1,6,7]]
        elif index == 5:
            self.obs = [obs_list[i] for i in [2,3,4,5]]
        elif index == 6:
            self.obs = [obs_list[i] for i in [2,3,6,7]]
        else: print("The index for initializing the env must be between 1 and 6!")


    def reset(self):
        self.time_log = 0                          # Reset time
        for i in range(self.obs_num):              # Clear the path record array for all obstacles
            self.path[i] = np.array([[]]).reshape(-1, 3)

    def update_obs_pos(self,uav_pos):
        """Update all obstacle positions and determine which one is closest to the UAV, returning its dic."""
        distance = np.inf
        temp_dic = None
        for i in range(self.obs_num):
            self.time_log, dic = self.obs[i](self.time_log, self.time_step)
            self.path[i] = np.vstack((self.path[i], dic['obsCenter']))
            self.time_log -= self.time_step  # The obs function has its own time_log increment, so we need to subtract it here.

            if self.distanceCost(dic['obsCenter'],uav_pos)-self.obs_r[i] < distance:
                distance = self.distanceCost(dic['obsCenter'],uav_pos)
                temp_dic = dic
                temp_dic['obs_r'] = self.obs_r[i]
                temp_dic['d'] = distance - self.obs_r[i]
        self.time_log += self.time_step

        return temp_dic

    def calDynamicState(self, uav_pos, obs_center, obs_R, v_obs):
        """State obtained from the reinforcement learning model."""
        s1 = (obs_center - uav_pos)*(self.distanceCost(obs_center,uav_pos)-obs_R)/self.distanceCost(obs_center,uav_pos)
        s2 = self.goal - uav_pos
        s3 = v_obs
        return np.append(s1,[s2,s3])

    def calRepulsiveMatrix(self, uavPos, obsCenter, obsR, row0):
        n = self.partialDerivativeSphere(obsCenter, uavPos, obsR)
        tempD = self.distanceCost(uavPos, obsCenter) - obsR
        row = row0 * np.exp(1-1/(self.distanceCost(uavPos,self.goal)*tempD))
        T = self.calculateT(obsCenter, uavPos, obsR)
        repulsiveMatrix = np.dot(-n,n.T) / T**(1/row) / np.dot(n.T,n)[0][0]
        return repulsiveMatrix

    def calTangentialMatrix(self, uavPos, obsCenter, obsR, theta, sigma0):
        n = self.partialDerivativeSphere(obsCenter, uavPos, obsR)
        T = self.calculateT(obsCenter, uavPos, obsR)
        partialX = (uavPos[0] - obsCenter[0]) * 2 / obsR ** 2
        partialY = (uavPos[1] - obsCenter[1]) * 2 / obsR ** 2
        partialZ = (uavPos[2] - obsCenter[2]) * 2 / obsR ** 2
        tk1 = np.array([partialY, -partialX, 0],dtype=float).reshape(-1,1)
        tk2 = np.array([partialX*partialZ, partialY*partialZ, -partialX**2-partialY**2],dtype=float).reshape(-1,1)
        originalPoint = np.array([np.cos(theta), np.sin(theta), 0]).reshape(1,-1)
        tk = self.trans(originalPoint, tk1.squeeze(), tk2.squeeze(), n.squeeze())
        tempD = self.distanceCost(uavPos, obsCenter) - obsR
        sigma = sigma0 * np.exp(1-1/(self.distanceCost(uavPos,self.goal)*tempD))
        tangentialMatrix = tk.dot(n.T) / T**(1/sigma) / self.calVecLen(tk.squeeze()) / self.calVecLen(n.squeeze())
        return tangentialMatrix
    def getqNext(self, uavPos, obsCenter, vObs, obsR, row0, sigma0, theta, qBefore):
        u = self.initField(uavPos, self.V0, self.goal)
        repulsiveMatrix = self.calRepulsiveMatrix(uavPos, obsCenter, obsR, row0)
        tangentialMatrix = self.calTangentialMatrix(uavPos, obsCenter, obsR, theta, sigma0)
        T = self.calculateT(obsCenter, uavPos, obsR)
        vp = np.exp(-T / self.lam) * vObs
        M = np.eye(3) + repulsiveMatrix + tangentialMatrix
        ubar = (M.dot(u - vp.reshape(-1, 1)).T + vp.reshape(1, -1)).squeeze()
        # Limit the magnitude of ubar to avoid abrupt trajectory changes after entering the obstacle.
        if self.calVecLen(ubar) > 5:
            ubar = ubar/self.calVecLen(ubar)*5
        if qBefore[0] is None:
            uavNextPos = uavPos + ubar * self.step_size
        else:
            uavNextPos = uavPos + ubar * self.step_size
            _, _, _, _, qNext = self.kinematicConstrant(uavPos, qBefore, uavNextPos)
        return uavNextPos

    def kinematicConstrant(self, q, qBefore, qNext):
        """
        Kinematic constraint function: returns (previous heading angle, previous climb angle, constrained heading angle, constrained climb angle, constrained next position qNext)
        """
        # Calculate the trajectory angle x1 and climb angle gam1 from qBefore to q
        qBefore2q = q - qBefore
        if qBefore2q[0] != 0 or qBefore2q[1] != 0:
            x1 = np.arcsin(np.abs(qBefore2q[1] / np.sqrt(qBefore2q[0] ** 2 + qBefore2q[1] ** 2)))  # 这里计算的角限定在了第一象限的角 0-pi/2
            gam1 = np.arcsin(qBefore2q[2] / np.sqrt(np.sum(qBefore2q ** 2)))
        else:
            return None, None, None, None, qNext
        # Calculate the trajectory angle x2 and climb angle gam2 from q to qNext
        q2qNext = qNext - q
        x2 = np.arcsin(np.abs(q2qNext[1] / np.sqrt(q2qNext[0] ** 2 + q2qNext[1] ** 2)))  # 这里同理计算第一象限的角度
        gam2 = np.arcsin(q2qNext[2] / np.sqrt(np.sum(q2qNext ** 2)))

        # Calculate the angle of the vector with respect to the positive x-axis (0 to 2 × π) based on its quadrant.
        if qBefore2q[0] > 0 and qBefore2q[1] > 0:
            x1 = x1
        if qBefore2q[0] < 0 and qBefore2q[1] > 0:
            x1 = np.pi - x1
        if qBefore2q[0] < 0 and qBefore2q[1] < 0:
            x1 = np.pi + x1
        if qBefore2q[0] > 0 and qBefore2q[1] < 0:
            x1 = 2 * np.pi - x1
        if qBefore2q[0] > 0 and qBefore2q[1] == 0:
            x1 = 0
        if qBefore2q[0] == 0 and qBefore2q[1] > 0:
            x1 = np.pi / 2
        if qBefore2q[0] < 0 and qBefore2q[1] == 0:
            x1 = np.pi
        if qBefore2q[0] == 0 and qBefore2q[1] < 0:
            x1 = np.pi * 3 / 2


        # Calculate the angle of the vector with respect to the positive x-axis (0 to 2 × π) based on its quadrant.
        if q2qNext[0] > 0 and q2qNext[1] > 0:
            x2 = x2
        if q2qNext[0] < 0 and q2qNext[1] > 0:
            x2 = np.pi - x2
        if q2qNext[0] < 0 and q2qNext[1] < 0:
            x2 = np.pi + x2
        if q2qNext[0] > 0 and q2qNext[1] < 0:
            x2 = 2 * np.pi - x2
        if q2qNext[0] > 0 and q2qNext[1] == 0:
            x2 = 0
        if q2qNext[0] == 0 and q2qNext[1] > 0:
            x2 = np.pi / 2
        if q2qNext[0] < 0 and q2qNext[1] == 0:
            x2 = np.pi
        if q2qNext[0] == 0 and q2qNext[1] < 0:
            x2 = np.pi * 3 / 2

        # Constrain heading angle x   
        deltax1x2 = self.angleVec(q2qNext[0:2], qBefore2q[0:2])  
        if deltax1x2 < self.xmax:
            xres = x2
        elif x1 - x2 > 0 and x1 - x2 < np.pi:  # Note these logic
            xres = x1 - self.xmax
        elif x1 - x2 > 0 and x1 - x2 > np.pi:
            xres = x1 + self.xmax
        elif x1 - x2 < 0 and x2 - x1 < np.pi:
            xres = x1 + self.xmax
        else:
            xres = x1 - self.xmax

        # Constrain climb angle gam   Note: The climb angle is only discussed in the range of -pi/2 to pi/2, which happens to be the same as the range of arcsin.  gamres is the constrained climb angle
        if np.abs(gam1 - gam2) <= self.gammax:
            gamres = gam2
        elif gam2 > gam1:
            gamres = gam1 + self.gammax
        else:
            gamres = gam1 - self.gammax
        if gamres > self.maximumClimbingAngle:
            gamres = self.maximumClimbingAngle
        if gamres < self.maximumSubductionAngle:
            gamres = self.maximumSubductionAngle

        # Calculate the coordinates of the constrained next point qNext
        Rq2qNext = self.distanceCost(q, qNext)
        deltax = Rq2qNext * np.cos(gamres) * np.cos(xres)
        deltay = Rq2qNext * np.cos(gamres) * np.sin(xres)
        deltaz = Rq2qNext * np.sin(gamres)

        qNext = q + np.array([deltax, deltay, deltaz])
        return x1, gam1, xres, gamres, qNext

    def initField(self, pos, V0, goal):
        """Calculate initial flow field and return column vector."""
        temp1 = pos[0] - goal[0]
        temp2 = pos[1] - goal[1]
        temp3 = pos[2] - goal[2]
        temp4 = self.distanceCost(pos, goal)
        return -np.array([temp1, temp2, temp3], dtype=float).reshape(-1, 1) * V0 / temp4

    def calPathLen(self, path):
        """ Calculate the length of a trajectory."""
        num = path.shape[0]
        len = 0
        for i in range(num - 1):
            len += self.distanceCost(path[i, :], path[i + 1, :])
        return len

    def trans(self, originalPoint, xNew, yNew, zNew):
        """
        Coordinate transformation to Earth coordinates
        newX, newY, newZ are the direction vectors on the three axes in the new coordinate system
        Returns a column vector
        """
        lenx = self.calVecLen(xNew)
        cosa1 = xNew[0] / lenx
        cosb1 = xNew[1] / lenx
        cosc1 = xNew[2] / lenx

        leny = self.calVecLen(yNew)
        cosa2 = yNew[0] / leny
        cosb2 = yNew[1] / leny
        cosc2 = yNew[2] / leny

        lenz = self.calVecLen(zNew)
        cosa3 = zNew[0] / lenz
        cosb3 = zNew[1] / lenz
        cosc3 = zNew[2] / lenz

        B = np.array([[cosa1, cosb1, cosc1],
                      [cosa2, cosb2, cosc2],
                      [cosa3, cosb3, cosc3]],dtype=float)

        invB = np.linalg.inv(B)
        return np.dot(invB, originalPoint.T)

    def save_data(self):
        np.savetxt('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/multi_csv/start.csv', self.start, delimiter=',')
        np.savetxt('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/multi_csv/goal.csv', self.goal, delimiter=',')
        np.savetxt('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/multi_csv/obs_r_list.csv',self.obs_r,delimiter=',')
        for i in range(self.obs_num):
            np.savetxt('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/multi_csv/obs{}_trace.csv'.format(i),self.path[i],delimiter=',')


    @staticmethod
    def distanceCost(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    @staticmethod
    def angleVec(vec1, vec2):  # Calculate the angle between two vectors
        temp = np.dot(vec1, vec2) / np.sqrt(np.sum(vec1 ** 2)) / np.sqrt(np.sum(vec2 ** 2))
        temp = np.clip(temp, -1, 1)  # Clip to handle potential numerical errors
        theta = np.arccos(temp)
        return theta


    @staticmethod
    def partialDerivativeSphere(obs, pos, r):
        """Calculate the partial derivative of the spherical obstacle equation and return a column vector."""
        temp1 = pos[0] - obs[0]
        temp2 = pos[1] - obs[1]
        temp3 = pos[2] - obs[2]
        return np.array([temp1,temp2,temp3],dtype=float).reshape(-1,1)*2/r**2


    @staticmethod
    def calculateT(obs, pos, r):
        """Calculate T."""
        temp1 = pos[0] - obs[0]
        temp2 = pos[1] - obs[1]
        temp3 = pos[2] - obs[2]
        return (temp1**2 + temp2**2 + temp3**2)/r**2

    @staticmethod
    def calVecLen(vec):
        """Calculate the length of a vector."""
        return np.sqrt(np.sum(vec**2))

    @staticmethod
    def load_model(method):
        import sys
        sys.path.append('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-{}-random_start'.format(method)) # Solve the problem that the saved model path is not the same as the loaded file path
 
        dynamicController = torch.load('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-{}-random_start/TrainedModel/dynamicActor.pkl'.format(method), map_location=device)
        return dynamicController
    
    @staticmethod
    def load_human(method):
        import sys
        sys.path.append('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-{}-random_start'.format(method)) # Solve the problem that the saved model path is not the same as the loaded file path
        
        humanController = torch.load('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-{}-random_start/TrainedModel/dynamicActor_h.pkl'.format(method), map_location=device)
        return humanController


if __name__ == "__main__":
    from Method import transformAction  # Place it here to avoid circular import
    from Method import get_reward_multiple
    from Method import drawActionCurve
    env = Environment(3)              # Initialize different dynamic scenarios with different numbers
    config = Config()
    METHOD = 'DDPG'                       # Fill in the model to be tested here
    controller = env.load_model(METHOD)
    human=env.load_human(METHOD)
    uav_pos = env.start
    uav_path = env.start.reshape(1,-1)   # Record UAV trace array
    actionCurve = np.array([])
    action_trace = np.array([]).reshape(-1,3)  # Record action trajectory array (three actions just fit n*3 shape)
    d_list = np.array([])                      # Store the distance to the threat surface
    qBefore = [None,None,None]
    reward_sum = 0
    reward_stack=[]
    if_test_origin_ifds = False        # Whether to use the untrained ifds algorithm
    threat_index = 0
    # Calculate the parameters needed for uncertainty
    action_matrix=np.zeros((3, 3))
    num_runs = 10
    tau = 1.0
    I_3 = np.eye(3)
    for step in range(500):
        data_dic = env.update_obs_pos(uav_pos)
        v_obs, obs_center, obs_R = data_dic['v'], data_dic['obsCenter'], data_dic['obs_r']
        if if_test_origin_ifds:
            a = [1,1.5,1.5]        # Default ifds action values
        else:
            state = env.calDynamicState(uav_pos,obs_center,obs_R,v_obs)
            state = torch.as_tensor(state, dtype=torch.float, device=device)
            # Switch model to evaluation mode
            action_matrix.fill(0)
            controller.eval()
            action_sum = np.zeros(config.act_dim)
            for _ in range(num_runs):
                 action = controller(state).cpu().detach().numpy()
                 action_sum += action
                 action_matrix+=np.dot(action.T,action)
            a = action_sum / num_runs 
            a = transformAction(a, config.actionBound, config.act_dim)
            # Calculate uncertainty
            Var1=np.dot(np.array(a).T,np.array(a))
            Var2=(1/tau)*I_3+(1/num_runs)* action_matrix
            Var3=Var2-Var1
            Uncertainty=abs(np.max(np.diagonal(Var3).flatten()))
            print(Uncertainty)
            # Action selection
            if Uncertainty>7:
                a=human(state).cpu().detach().numpy()
                a=transformAction(a, config.actionBound, config.act_dim)

            action_trace = np.append(action_trace,np.array(a).reshape(-1,3),axis=0)  # Save action trace
            actionCurve = np.append(actionCurve, a)

        # Threat index calculation
        if data_dic['d'] <= 0:
            threat_index = np.inf
        elif data_dic['d'] <= 0.4:
            threat_index += 1/data_dic['d']
        d_list = np.append(d_list,data_dic['d'])

        uav_next_pos = env.getqNext(uav_pos,obs_center,v_obs,obs_R,a[0],a[1],a[2],qBefore)
        if env.distanceCost(uav_next_pos,obs_center)<=obs_R:
            print("Collision detected!")
        r= get_reward_multiple(env,uav_next_pos,data_dic)
        reward_stack.append(r)
        reward_sum += r
        qBefore = uav_pos
        uav_pos = uav_next_pos
        if env.distanceCost(uav_pos,env.goal)<env.threshold:
            uav_path = np.vstack((uav_path, env.goal))
            _ = env.update_obs_pos(uav_pos)
            break
        uav_path = np.vstack((uav_path, uav_pos))
        
    drawActionCurve(actionCurve.reshape(-1,3))
    print("Path length: {}, Path reward: {}, Threat index: {}".format(env.calPathLen(uav_path),reward_sum,threat_index))

    plt.show()


