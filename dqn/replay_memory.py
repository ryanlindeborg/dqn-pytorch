import random
import torch
from collections import deque
from experience import Experience

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
            raise Exception(f"Not enough experiences in memory buffer ({len(self.memory)}) to sample from, for batch size of {batch_size}")

        replay_sample = random.sample(self.memory, batch_size)
        replay_states, replay_actions, replay_rewards, replay_next_states = self.extract_replay_tensors_from_sample_batch(replay_sample)

        return replay_states, replay_actions, replay_rewards, replay_next_states

    def extract_replay_tensors_from_sample_batch(self, batch_experiences):
        # Will convert batch of individual experiences to one experience whose attributes represent tuple of all experiences together
        # E.g., state tuple attribute is n-dimensional instead of 1-dimensional
        # Zip here takes in an iterable of experiences
        # * operator is used to unpack the namedtuple into separate positional arguments
        batch_experience = Experience(*zip(*batch_experiences))

        # Takes tuple of each property from all experiences, and
        # concatenates into individual tensor for state, action, reward, next state
        replay_tensor_state = torch.cat(batch_experience.state)
        replay_tensor_action = torch.cat(batch_experience.action)
        replay_tensor_reward = torch.cat(batch_experience.reward)
        replay_tensor_next_state = torch.cat(batch_experience.next_state)

        # Reshape state tensors as each state is 4-dimensional
        replay_tensor_state = torch.reshape(replay_tensor_state, (-1, 4))
        replay_tensor_next_state = torch.reshape(replay_tensor_next_state, (-1, 4))

        return (replay_tensor_state, replay_tensor_action, replay_tensor_reward, replay_tensor_next_state)

