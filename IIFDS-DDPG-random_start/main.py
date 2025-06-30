import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from DDPGModel import *
from IIFDS import IIFDS
from Method import getReward, transformAction, setup_seed, test, test_multiple
from draw import Painter
import matplotlib.pyplot as plt
import random
from config import Config
import os


if __name__ == "__main__":
    setup_seed(5)   # Set the random seed.

    conf = Config()
    iifds = IIFDS()
    obs_dim = conf.obs_dim
    act_dim = conf.act_dim

    dynamicController = DDPG(obs_dim, act_dim)
    if conf.if_load_weights and \
       os.path.exists('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/ac_weights.pkl') and \
       os.path.exists('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/ac_tar_weights.pkl'):
        dynamicController.ac.load_state_dict(torch.load('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/ac_weights.pkl'))
        dynamicController.ac_targ.load_state_dict(torch.load('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/ac_tar_weights.pkl'))
    
    actionBound = conf.actionBound

    MAX_EPISODE = conf.MAX_EPISODE
    MAX_STEP = conf.MAX_STEP
    update_every = conf.update_every
    batch_size = conf.batch_size
    noise = conf.noise
    update_cnt = 0
    rewardList = {1:[],2:[],3:[],4:[],5:[],6:[]}        # Record rewards for each test environment
    maxReward = -np.inf

    for episode in range(MAX_EPISODE):
        q = iifds.start + np.random.random(3)*3  # Random starting position to diversify training scenarios.
        qBefore = [None, None, None]
        iifds.reset()
        for j in range(MAX_STEP):
            dic = iifds.updateObs()
            vObs, obsCenter, obsCenterNext = dic['v'], dic['obsCenter'], dic['obsCenterNext']
            obs = iifds.calDynamicState(q, obsCenter)
            if episode > 50:
                noise *= 0.99995
                if noise <= 0.1: noise = 0.1
                action = dynamicController.get_action(obs, noise_scale=noise)
            else:
                action = [random.uniform(-1,1) for i in range(act_dim)]        # Here the output layer of the actor is tanh
            # Interact with the environment
            actionAfter = transformAction(action, actionBound, act_dim)  # Linearly map -1 to 1 to the corresponding action values
            qNext = iifds.getqNext(q, obsCenter, vObs, actionAfter[0], actionAfter[1], actionAfter[2], qBefore)
            obs_next = iifds.calDynamicState(qNext, obsCenterNext)
            reward = getReward(obsCenterNext, qNext, q, qBefore, iifds)

            done = True if iifds.distanceCost(iifds.goal, qNext) < iifds.threshold else False
            dynamicController.replay_buffer.store(obs, action, reward, obs_next, done)

            if episode >= 50 and j % update_every == 0:
                if dynamicController.replay_buffer.size >= batch_size:
                    update_cnt += update_every
                    for _ in range(update_every):
                        batch = dynamicController.replay_buffer.sample_batch(batch_size)
                        dynamicController.update(data=batch)
            if done: break
            qBefore = q
            q = qNext
        testReward = test_multiple(dynamicController.ac.pi,conf)
        print('Episode:', episode, 'Reward1:%2f' % testReward[0], 'Reward2:%2f' % testReward[1],
              'Reward3:%2f' % testReward[2], 'Reward4:%2f' % testReward[3],
              'Reward5:%2f' % testReward[4], 'Reward6:%2f' % testReward[5],
              'average reward:%2f' % np.mean(testReward), 'update_cnt:%d' % update_cnt)
        for index, data in enumerate(testReward):
            rewardList[index + 1].append(data)
        if episode > MAX_EPISODE / 2:
            if np.mean(testReward) > maxReward:
                maxReward = np.mean(testReward)
                print('Current episode cumulative average reward is the best in history, model saved!')
                torch.save(dynamicController.ac.pi, '/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/dynamicActor.pkl')
                torch.save(dynamicController.ac.q, '/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/dynamicCritic.pkl')

                # Save weights for transfer learning in various scenarios
                torch.save(dynamicController.ac.state_dict(), '/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/ac_weights.pkl')
                torch.save(dynamicController.ac_targ.state_dict(), '/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/TrainedModel/ac_tar_weights.pkl')

    # Plotting
    for index in range(1, 7):
        # painter = Painter(load_csv=True, load_dir='F:/MasterDegree/毕业设计/实验数据/figure_data_d{}.csv'.format(index))
        painter = Painter(load_csv=True, load_dir='/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/selfdataddpg//figure_data_d{}.csv'.format(index))
        painter.addData(rewardList[index], 'IIFDS-DDPG')
        painter.saveData('/home/prolee/apps/UAV_Obstacle_Avoiding_DRL-master/Dynamic_obstacle_avoidance/IIFDS-DDPG-random_start/selfdataddpg//figure_data_d{}.csv'.format(index))
        painter.drawFigure()






