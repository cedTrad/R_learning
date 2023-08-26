import os
import gym
import numpy as np
import pandas as pd 
import time

#env = gym.make("CartPole-v1", render_mode = "human")
env = gym.make('Blackjack-v1', render_mode = "human")


def policy(state):
    return 0 if state[0] > 15 else 1


def generate_episode(policy):
    
    episode = []
    state = env.reset()[0]
    num_timestep = 100
    treward = 0
    
    for i in range(num_timestep):        
        action = policy(state)
        next_state, reward, done, info, _ = env.step(action)
        
        episode.append(
            (state, action, reward)
        )
        env.render()
        if done:
            break
        
        treward+=reward
        state = next_state
        
    print(f"step={i}   treward={treward}")
    


generate_episode(policy)

