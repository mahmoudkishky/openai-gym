from collections import deque
import random
import config
import sys 

class ReplayBuffer():

    def __init__(self):

        self.replay_buffer = deque(maxlen = config.MAX_REPLAY_BUFFER_SIZE)
        #sys.stdout.flush()

    def get_size(self):

        return len(self.replay_buffer)
    
    def add_item(self, experience):

        current_state, action, reward, next_state, done = experience
        self.replay_buffer.append((current_state, action, reward, next_state, done))

    def sample(self, batch_size):

        return random.sample(self.replay_buffer, batch_size)