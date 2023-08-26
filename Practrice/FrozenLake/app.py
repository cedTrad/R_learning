import os
import gym
import time
import numpy as np
import pandas as pd 

from iterations import value_iteration, extract_policy, policy_iteration

ENV_NAME = "FrozenLake8x8-v1"
GAMMA = 0.9
TEST_EPISODES = 20

#n=

class Agent:
    
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(
            collections.Counter)
        self.values = collections.defaultdict(float)
        
    
    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state
            
            
    
    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            val = reward + GAMMA * self.values[tgt_state]
            action_value += (count / total) * val
        return action_value
        
    
    
    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action
    
    
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            next_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_return
    
    
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [
                self.calc_action_value(state, action) 
                for action in range(self.env.action_space.n)
            ]
            self.values[state] = max(state_values)
            
    



env = gym.make(ENV_NAME, render_mode = "human")
state = env.reset()[0]
env.render()


value_table = value_iteration(env)
policy = extract_policy(value_table, env)

policy2 = policy_iteration(env)

x = np.all(policy2 == policy)
print(x)

max_iterations = 500

def my_policy(state):
    return int(policy[state])
    

for i in range(max_iterations):
    #action = env.action_space.sample()
    action = my_policy(state)
    time.sleep(0.1)
    new_state, reward, done, info, prob = env.step(action)
    print(f"step={i}    new_state : {new_state}   reward:{reward} done={done}")
    env.render()
    if done:
        break
    
    state = new_state
    