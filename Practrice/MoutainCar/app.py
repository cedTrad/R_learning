import gym
import os
from matplotlib import pyplot as plt

env = gym.make('MountainCar-v0')
env._max_episode_steps = 1000

env.reset()
env.render()

print(f"Position ranges from : {env.min_position} to {env.max_position}")
print(f"Velocity ranges from : {-env.max_speed} to {env.max_speed}")

print(f"max step {env._max_episode_steps}")


class RandomAgent:
    
    def __init__(self, env):
        self.env = env
        



class Agent:
    
    def __init__(self, env):
        self.env = env
        
        
    
    