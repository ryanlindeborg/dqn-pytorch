import gym
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .experience import Experience

class DQN(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()

        # Input dimensions of state
        # This can be the input dimensions of an image, or any other state format
        self.input_dims = input_dims
        # Output dimensions are number of actions that can be taken for a given state
        self.output_dims = output_dims

        # This network consists of 2 fully connected layers with RELU nonlinearity and the otput layer
        hidden_layer_1_size = 256
        hidden_layer_2_size = 128
        self.fc1_layer = nn.Linear(in_features=self.input_dims, out_features=hidden_layer_1_size)
        self.fc2_layer = nn.Linear(in_features=hidden_layer_1_size, out_features=hidden_layer_2_size)
        self.output_layer = nn.Linear(in_features=hidden_layer_2_size, out_features=self.output_dims)

    def forward(self, input_tensor):
        print(f"Input tensor size: {input_tensor.size()}")
        input_tensor = input_tensor.flatten(start_dim=1)
        print(f"Input tensor size after flattening: {input_tensor.size()}")
        output_tensor = F.relu(self.fc1_layer(input_tensor))
        output_tensor = F.relu(self.fc2_layer(output_tensor))
        output_tensor = self.output_layer(output_tensor)
        return output_tensor

def run_dqn_on_cartpole():
    pass

if __name__ == "__main__":
    run_dqn_on_cartpole()
