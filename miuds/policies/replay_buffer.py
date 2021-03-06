from collections import deque
import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, capacity, transition):
        self.keys = keys
        self.memory = deque(maxlen=capacity)
        self.transition = transition

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        raw_batch = random.sample(self.memory, batch_size)
        batch = [np.vstack(b) for b in zip(*raw_batch)]
        batch = self.transition(*batch)
        return batch

    def __len__(self):
        return len(self.memory)
