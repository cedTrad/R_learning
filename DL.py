import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2

from tensorflow.keras.layers import SimpleRNN, LSTM

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    

def cw(target):
    c0, c1 = np.bincount(target)
    w0 = (1/c0) * (len(target))/2
    w1 = (1/c1) * (len(target))/2
    return {0 : w0, 1 : w1}



optimizer = Adam(learning_rate = 0.001)

# Dense model
def create_DNN_model(ent_dim, hl=1, hu=128, optimizer=optimizer, dropout=False, regularize=False, reg=l1(0.0005)):
    if not regularize:
        reg = None
        
    model = Sequential()
    model.add(
        Dense(hu, activation='relu', input_dim=ent_dim,
              activity_regularizer = reg)
    )
    for _ in range(hl):
        model.add(
            Dense(hu, activation='relu',
                  activity_regularizer = reg)
        )
        if dropout:
            model.add(
                Dropout(rate, seed = 100)
            )
    model.add(
        Dense(1, activation = 'sigmoid')
    )
    model.compile(loss = 'binary_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy']
                  )
    return model




