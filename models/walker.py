import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.nn import softmax

class Actor(Model):
    
    def __init__(self, action_dim):
        super().__init__()

        self.fc1 = Dense(256, activation='relu')
        self.fc2 = Dense(256, activation='relu')
        self.mu = Dense(action_dim, activation=None)
        self.log_std = Dense(action_dim, activation=None)


    def call(self, state):

        x = self.fc1(state)
        x = self.fc2(x)
        mu = self.mu(x)
        log_std = self.log_std(x)

        log_std = tf.clip_by_value(log_std, -20, 2)
        std = tf.exp(log_std) + 1e-8

        return mu, std


class Critic(Model):
    
    def __init__(self):
        super().__init__()

        self.fc1 = Dense(256, activation='relu')
        self.fc2 = Dense(256, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, actor_action):

        input = tf.concat([state, actor_action], axis = -1)
        x = self.fc1(input)
        x = self.fc2(x)
        q_value = self.q(x)
        
        return q_value
