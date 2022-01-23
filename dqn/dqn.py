import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # print(f"Input tensor size: {input_tensor.size()}")
        # input_tensor = input_tensor.flatten(start_dim=1)
        # print(f"Input tensor size after flattening: {input_tensor.size()}")
        input_tensor = input_tensor.to(self.device)
        output_tensor = F.relu(self.fc1_layer(input_tensor))
        output_tensor = F.relu(self.fc2_layer(output_tensor))
        output_tensor = self.output_layer(output_tensor)
        return output_tensor

    def get_current_q_values(self, states, actions):
        # print(f"Actions dimensions: {actions.size()}")
        return self(states).gather(dim=1, index=actions.unsqueeze(-1))

    def get_next_q_values(self, next_states):
        # TODO: Only compute next q values on non-final states, to reduce extra work
        # State is final if max value of state is 0
        # print(f"Next states dims: {next_states.size()}")
        # final_state_mask = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        # non_final_state_mask = (final_state_mask == False)
        # non_final_states = next_states[non_final_state_mask]

        # batch_size = next_states.shape[0]
        # First initialize q values to 0 then populate values in non-final states
        # next_q_values = torch.zeros(batch_size).to(self.device)
        # TODO: Update this to use target network if doing double DQN
        # next_q_values[non_final_state_mask] = self(non_final_states).max(dim=1)[0].detach()
        next_q_values = self(next_states).max(dim=1)[0].detach()
        return next_q_values
