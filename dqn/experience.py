# This class represents one experience of a given transition in reinforcement learning
# More concretely, this experience will hold information on current state and action,
# and reward and next state from executing that action
from collections import namedtuple
Experience = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

#TODO: Refactor experience to be separate class, and handle zipping logic with objects
# Maybe take approach from Machine Learning with Phil where set index of memory when add it, so that can take index to get tensor back

# class Experience():
#     def __init__(self, state, action, reward, next_state):
#         self.state = state
#         self.action = action
#         self.reward = reward
#         self.next_state = next_state
