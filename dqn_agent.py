import random
import torch

class DQNAgent():
    def __init__(self, exploration_strategy, num_actions, device):
        self.exploration_strategy = exploration_strategy
        self.num_actions = num_actions
        self.device = device

    # Given a state, returns the greedy action with (1-epsilon) probability and a random action otherwise
    def get_action(self, state, dqn):
        epsilon = self.exploration_strategy.get_exploration_rate()
        if random.random() <= epsilon:
            action = self.select_random_action()
            action = torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                logits = dqn(state)
                print(f"Feedforward through dqn to get action, shape: {logits.size()}")
                print(f"Logits: {logits}")
                print(f"Argmax: {logits.argmax(dim=1)}")
                action = dqn(state).argmax(dim=1).to(self.device)

        # Now that you have selected the action, decay the exploration rate so that you are more likely to select greedy action
        self.exploration_strategy.decay_exploration_rate()

        return action

    def select_random_action(self):
        # For now, this just returns an integer within range of 0 to num_actions - 1 for the selected action (cartpole only has two actions)
        return random.randint(0, self.num_actions - 1)