import numpy as np 
import pandas as pd
import time
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, RMSprop

from collections import deque
#from IPython.display import clear_output


class DQLAgent:
    
    def __init__(self, env, gamma=0.95, hu=24, opt=Adam, lr=0.001, finish=False, render=False):
        self.finish = finish
        self.env = env
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = gamma
        self.batch_size = 32
        self.max_treward = 0
        self.averages = list()
        self.memory = deque(maxlen=2000)
        self.osn = env.observation_space.shape[0]
        self.model = self._build_model(hu, opt, lr)
        
    def _build_model(self, hu, opt, lr):
        model = Sequential()
        model.add(Dense(hu, input_dim = self.osn, activation="relu"))
        model.add(Dense(hu, activation="relu"))
        model.add(Dense(self.env.action_space.n, activation="linear"))
        model.compile(loss="mse", optimizer=opt(learning_rate=lr))
        return model
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()
        action = self.model.predict(state, verbose=0)[0]
        return np.argmax(action)
    
    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if not done:
                reward += self.gamma * np.amax(
                    self.model.predict(next_state, verbose=0)[0]
                )
                target = self.model.predict(state, verbose=0)
                target[0, action] = reward
                self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_min
    
    def learn(self, episodes):
        trewards = []
        for e in range(1, episodes+1):
            state = self.env.reset()
            state = np.reshape(state, [1, self.osn])
            for i in range(500):
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.osn])
                
                self.memory.append([state, action, reward, next_state, done])
                
                state = next_state
                av = 0
                if done:
                    #clear_output(True)
                    treward = i + 1
                    trewards.append(treward)
                    av = sum(trewards[-25:]) / 25
                    self.averages.append(av)
                    self.max_treward = max(self.max_treward, treward)
                    templ = "episode : {:4d}/{} | treward : {:4d} |"
                    templ += "av : {:6.1f} | max : {:4d}"
                    print(templ.format(e, episodes, treward, av, self.max_treward), end="\r")
                    break
            
            if av > 195 and self.finish:
                break
            
            if len(self.memory) > self.batch_size:
                self.replay()
                
    
    def test(self, episodes):
        trewards = []
        for e in range(1, episodes+1):
            state = self.env.reset()
            for i in range(500):
                state = np.reshape(state, [1, self.osn])
                action = np.argmax(self.model.predict(state, verbose=0)[0])
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                if done:
                    treward = i+1
                    trewards.append(treward)
                    print("episode : {:4d}/{}  | treward : {:4d}".format(e, episodes, treward), end="\r")
                    break
        return trewards








class FQLAgent:
    
    def __init__(self, hidden_units, learning_rate, learn_env, valid_env):
        self.learn_env = learn_env
        self.valid_env = valid_env
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.98
        self.learning_rate = learning_rate
        self.gamma = 0.95
        self.batch_size = 128
        self.max_treward = 0
        self.trewards = list()
        self.averages = list()
        self.performances = list()
        self.aperformances = list()
        self.vperformances = list()
        self.memory = deque(maxlen = 2000)
        self.model = self._build_model(hidden_units, learning_rate)
        
    def _build_model(self, hu, lr):
        model = Sequential()
        model.add(
            Dense(hu, input_shape = (self.learn_env.lags, self.learn_env.n_features), activation="relu")
        )
        model.add(Dropout(0.3, seed = 100))
        model.add(Dense(hu, activation="relu"))
        model.add(Dropout(0.3, seed=100))
        model.add(
            Dense(2, activation="linear")
        )
        model.compile(loss = "mse", optmizer = RMSprop(lr = lr))
        
        return model
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return self.learn_env.action_space.sample()
        action = self.model.predict(state, verbose = 0)[0, 0]
        return np.argmax(action)
    
    def replay(self):
        batch = random.random(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if not done:
                reward += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0, 0])
            target = self.model.predict(state, verbose=0)
            target[0, 0, action] = reward
            self.model.fit(state, target, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def learn(self, episodes):
        for e in range(1, episodes+1):
            state = self.learn_env.reset()
            state = np.reshape(state, [1, self.learn_env.lags,
                                       self.learn_env.n_features])
            for i in range(1000):
                action = self.act(state)
                next_state, reward, done, info = self.learn_env.step(action)
                next_state = np.reshape(next_state, [1, self.learn_env.lags, self.learn_env.n_features])
                self.memory.append([state, action, reward, next_state, done])
                
                state = next_state
                
                if done:
                    treward = _ + 1
                    self.trewards.append(treward)
                    av = sum(self.trewards[-25:]) / 25
                    perf = self.learn_env.performance
                    self.averages.append(av)
                    self.performances.append(perf)
                    self.aperformances.append(
                        sum(self.performances[-25:])/25
                    )
                    
                    templ = "episode : {:2d}/{}  |  treward : {:4d} | "
                    templ += "perf : {:5.3f}  |  av : {:5.1f}  |  max : {:4d} "
                    print(templ.format(e, episodes, treward, perf, av, self.max_treward), end="\r")
                    
                    break
            
            self.validate(e, episodes)
            
            if len(self.memory) > self.batch_size:
                self.replay()
    
    def validate(self, e, episodes):
        state = self.valid_env.reset()
        state = np.reshape(state, [1, self.valid_env.lags,
                                   self.valid_env.n_features])
        
        for _ in range(1000):
            action = np.argmax(self.model.predict(state, verbose=0)[0, 0])
            next_state, reward, done, info = self.valid_env.step(action)
            state = np.reshape(next_state, [1, self.valid_env.lags,
                                            self.valid_env.n_features])
            
            if done:
                treward = _ + 1
                perf = self.valid_env.performance
                self.vperformances.append(perf)
                
                if e%20 == 0:
                    templ = 71 * '='
                    templ += '\nepisode : {:2d}/{}   VALIDATION | '
                    templ += 'treward : {:4d} | perf : {:5.3f} | '
                    templ += 'eps : {:.2f} \n'
                    templ += 71*'='
                    print(templ.format(e, episodes, treward, perf,
                                       self.epsilon))
                break
            
        
                
                