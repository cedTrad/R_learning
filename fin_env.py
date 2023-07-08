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
from IPython.display import clear_output


class observation_space:
    
    def __init__(self, n):
        self.shape = (n,)
        

class action_space:
    
    def __init__(self, n):
        self.n = n
    
    def seed(self, seed):
        pass
    
    def sample(self):
        return random.randint(0, self.n-1)
    
    


class Finance:
    
    def __init__(self, symbol, features, window, lags, leverage=1, min_performance=0.85,
                 start=0, end=None, mu=None, std=None):
        self.symbol = symbol
        self.features = features
        self.n_features = len(features)
        self.window = window
        self.lags = lags
        self.leverage = leverage
        self.min_performance = min_performance
        self.start = start
        self.end = end
        self.mu = mu
        self.std = std
        self.interval = "1d"
        self.observation_space = observation_space(self.lags)
        self.action_space = action_space(2)
        self._get_data()
        self._prepare_data()
        
        
    def _get_data(self):
        self.raw = Get_data(self.symbol, self.interval)
        
        
    def _prepare_data(self):
        self.data = pd.DataFrame(self.raw.close)
        self.data['r'] = np.log(self.data / self.data.shift(1))
        self.data.dropna(inplace = True)
        self.data['r'] = np.log(self.data / self.data.shift(1))
        self.data.dropna(inplace = True)
        self.data['s'] = self.data["close"].rolling(self.window).mean()
        self.data['m'] = self.data["r"].rolling(self.window).mean()
        self.data['v'] = self.data["r"].rolling(self.window).std()
        if self.mu is None:
            self.mu = self.data.mean()
            self.std = self.data.std()
        self.data_ = (self.data - self.data.mean()) / self.data.std()
        self.data_['d'] = np.where(self.data['r'] > 0, 1, 0)
        self.data_['d'] = self.data_['d'].astype(int)
        
        if self.end is not None:
            self.data = self.data.iloc[:self.end - self.start]
            self.data_ = self.data_.iloc[:self.end - self.start]
    
    
    def _get_state(self):
        return self.data[self.features].iloc[self.bar - self.lags : self.bar].values
    
    
    def seed(self, seed = None):
        random.seed(seed)
        np.random.seed(seed)
    
    def reset(self):
        self.treward = 0
        self.accurcy = 0
        self.bar = self.lags
        self.performance = 1
        state = self.data[self.features].iloc[self.bar - self.osn : self.bar]
        #state = self._get_state()
        return state.values
    
    def step(self, action):
        correct = action == self.data['d'].iloc[self.bar]
        ret = self.data['r'].iloc[self.bar] * self.leverage
        
        reward_1 = 1 if correct else 0
        reward_2 = abs(ret) if correct else -abs(ret)
        
        self.treward += reward_1
        self.bar += 1
        self.accurcy = self.treward / (self.bar - self.osn)
        
        self.performance *= math.exp(reward_2)
        
        if self.bar >= len(self.data):
            done = True
        elif reward == 1:
            done = False
        elif (self.performance < self.min_performance and self.bar > self.lags+5):
            done = True
        else:
            done = False
        
        state = self._get_state()
        info = {}
        return state, reward_1, reward_2*5, done, info

        

