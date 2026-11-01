import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.nn import softmax

class Agent(Model):
    
    def __init__(self, action_dim):
        super().__init__()

        self.fc1 = Dense(32, activation='relu')
        self.fc2 = Dense(32, activation='relu')
        #self.fc3 = Dense(64, activation='relu')
        self.mu = Dense(action_dim, activation='tanh')
        self.log_std = tf.Variable(tf.zeros(action_dim), trainable=True)

    def call(self, state):

        x = self.fc1(state)
        x = self.fc2(x)
        #x = self.fc3(x)
        mu = self.mu(x)
        std = tf.exp(self.log_std) + 1e-8
        return mu, std


