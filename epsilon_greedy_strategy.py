class EpsilonGreedyStrategy():
    def __init__(self, start_exploration_rate, min_exploration_rate, decay_rate):
        self.exploration_rate = start_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.decay_rate = decay_rate

    def get_exploration_rate(self):
        return self.exploration_rate

    def decay_exploration_rate(self):
        new_exploration_rate = self.exploration_rate * self.decay_rate
        # Decayed exploration rate cannot be less than minimum exploration rate
        self.exploration_rate = max(new_exploration_rate, self.min_exploration_rate)
