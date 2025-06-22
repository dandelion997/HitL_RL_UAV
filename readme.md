# README

This is a deep reinforcement learning project focused on autonomous obstacle avoidance algorithms for **UAV**. The project consists of two parts: obstacle avoidance in environments with a **single obstacle** and in environments with **multiple obstacles**. In dynamic environments, it adopts a combination of the **disturbed flow field algorithm** and a **single-agent reinforcement learning algorithm**.


## Dynamic environment

There are two methods to solve:

1. **DDPG**（In human-agent collaboration, the agent is based on the DDPG algorithm.）
2. **SAC**

DDPG converge fast. Though Soft Actor-Critic is an outstanding algorithm in DRL, it has no obvious effect in this environment.


## IFDS and IIFDS algorithm

This is an obstacle avoidance planning algorithm based on flow field, and the related code is named **IIFDS**.

## How to begin trainning

For example, you want to train the agent in dynamic environment with DDPG, what you need to do is just running the **main.py**, then **test.py**.

If you want to test the model in the environment with 4 obstacles, you just need to run  **Multi_obstacle_environment_test.py**.

## Requirements

numpy

torch

matplotlib

seaborn==0.11.1

## Files to illustrate

**calGs.m**: calculate the index Gs which shows the performance of the route.

**calLs.m**: calculate the index Ls which shows the performance of the route.

**draw.py**: this file includes the Painter class which can draw the reward curve of various methods.

**config.py**: this file give the setting of the parameters in trainning process of the algorithm such as the MAX_EPISODE, batch_size and so on.

**Method.py**: this file concludes many important methods such as how to calculate the reward of the agents.

**dynamic_obstacle_environment.py**: there are many dynamic obstacle environments' parameters in this file.

**Multi_obstacle_environment_test.py**: this file test the dynamic model in the environment in dynamic_obstacle_environment.py.


# A simple simulation example

- ![avatar](/Dynamic_obstacle_avoidance/GIF/compare_aifds.gif)



*all rights reserved.*

