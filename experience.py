# This class represents one experience of a given transition in reinforcement learning
# More concretely, this experience will hold information on current state and action,
# and reward and next state from executing that action
class Experience():
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
