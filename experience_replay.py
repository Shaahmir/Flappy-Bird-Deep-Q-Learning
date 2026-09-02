import random
from collections import deque
from typing import Tuple

class ReplayMemory:

    def __init__(self, capacity: int):
        self.memory = deque(maxlen = capacity)

    def append(self, transition: Tuple):
        self. memory.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)