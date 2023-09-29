import gym
import os
from matplotlib import pyplot as plt

env = gym.make('MountainCar-v0')
env._max_episode_steps = 1000

env.reset()
env.render()

