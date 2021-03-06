import random
import torch

class DQNAgent():
    def __init__(self, exploration_strategy, num_actions, device):
        self.exploration_strategy = exploration_strategy
        self.num_actions = num_actions
        self.device = device

    # Given a state, returns the greedy action with (1-epsilon) probability and a random action otherwise, if in training phase
    # Else takes greedy action
    def get_action(self, state, dqn, is_training=True):
        if is_training:
            epsilon = self.exploration_strategy.get_exploration_rate()
            if random.random() <= epsilon:
                action = self.select_random_action()
                action = torch.tensor([action], dtype=torch.long).to(self.device)
            else:
                # logits = dqn(state)
                # print(f"Feedforward through dqn to get action, shape: {logits.size()}")
                # print(f"Logits: {logits}")
                # action = dqn(state).argmax(dim=1).to(self.device)
                # action = dqn(state).argmax().to(self.device)
                action = dqn(state).argmax()
                action = torch.tensor([action]).to(self.device)

            # Now that you have selected the action, decay the exploration rate so that you are more likely to select
            # greedy action on next action selection
            # print(f"**************Epsilon exploration rate: {self.exploration_strategy.get_exploration_rate()}")
            self.exploration_strategy.decay_exploration_rate()
        else:
            # Not training anymore, so want to always take action with highest Q value
            with torch.no_grad():
                action = dqn(state).argmax()
                action = torch.tensor([action]).to(self.device)

        return action

    def select_random_action(self):
        # For now, this just returns an integer within range of 0 to num_actions - 1 for the selected action (cartpole only has two actions)
        return random.randint(0, self.num_actions - 1)

    def get_epsilon(self):
        return self.exploration_strategy.get_exploration_rate()