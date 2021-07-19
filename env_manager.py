import gym
import torch

from .experience import Experience

class EnvManager():
    def __init__(self, env_name, device, memory, dqn, agent=None):
        self.env_name = env_name
        # For now, just using environments from OpenAI Gym
        # To access certain properties of the OpenAI Gym environment, you have to unwrap it
        self.env = gym.make(self.env_name).unwrapped
        self.env.reset()
        self.done = False
        self.memory = memory
        # Batch size to sample replay memory for training
        self.replay_batch_size = 256
        self.dqn = dqn
        self.agent = agent
        self.device = device

    def get_num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        next_state, reward, self.done, info = self.env.step(action)
        return next_state, torch.tensor([reward], device=self.device)

    def close_env(self):
        self.env.close()

    def run_episode(self):
        num_steps = 0
        state = self.env.reset()
        while not self.done or num_steps < 1000:
            num_steps += 1
            # Select and take action
            action = self.agent.get_action(state=state, dqn=self.dqn)
            next_state, reward = self.take_action(action)

            # Add experience to memory
            self.memory.add_to_memory(Experience(state, action, reward, next_state))
            state = next_state

            # Learn on random batch from replay memory
            self.learn()

        print(f"Finished the episode after {num_steps} steps")



    def learn(self):
        try:
            replay_sample = self.memory.sample(self.replay_batch_size)


        except Exception as e:
            # This could occur if not enough experiences in replay buffer yet
            print(e)
