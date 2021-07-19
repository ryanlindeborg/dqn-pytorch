import random
from collections import deque

class ReplayMemory():
    def __init__(self, memory_capacity):
        self.memory_capacity = memory_capacity
        # Internal memory structure is a deque (https://docs.python.org/3/library/collections.html#collections.deque),
        # which is double-ended queue that supports thread-safe, memory efficient appends
        # and pops from either side of the deque with approximately the same O(1) performance in either direction
        self.memory = deque([], maxlen=self.memory_capacity)

    def add_to_memory(self, experience):
        if len(self.memory) == self.memory_capacity:
            # Remove oldest experience so can replace with newest experience
            self.memory.popleft()
        self.memory.append(experience)

    def sample(self, batch_size):
        # Only sample if have enough experiences in replay memory to make full sample of size batch_size
        if len(self.memory) < batch_size:
            raise Exception(f"Not enough experiences in memory buffer to sample from, for batch size of {batch_size}")

        return random.sample(self.memory, batch_size)