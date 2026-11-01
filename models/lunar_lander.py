import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.nn import softmax

class Agent(Model):
    
    def __init__(self, output_len):
        super().__init__()

        self.fc1 = Dense(64, activation='relu')
        self.fc2 = Dense(64, activation='relu')
        self.fc3 = Dense(64, activation='relu')
        self.out = Dense(output_len)

    def call(self, state):

        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        q = self.out(x)
        return q


