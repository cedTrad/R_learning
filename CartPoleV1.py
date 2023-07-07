import random
import gym
import time
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from agent import NNAgent

env = gym.make("CartPole-v1", render_mode = "human")

agent = NNAgent()

episodes = 100
agent.learn(episodes, env)

env.close()

sum(agent.scores) / len(agent.scores)
