import gym
import torch
import torch.nn.functional as F

from experience import Experience

class EnvManager():
    def __init__(self, env_name, device, memory, dqn, optimizer, gamma=1, agent=None):
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
        self.optimizer = optimizer
        self.gamma = gamma
        self.agent = agent
        self.device = device

    def get_num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        next_state, reward, self.done, info = self.env.step(action)
        return torch.tensor(next_state, dtype=torch.float32, device=self.device), torch.tensor([reward], device=self.device)

    def close_env(self):
        self.env.close()

    def run_episode(self):
        num_steps = 0
        # TODO: Can play with changing state to be from rendered screen instead of returned state from environment
        state = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        while not self.done or num_steps < 1000:
            num_steps += 1
            # Select and take action
            action = self.agent.get_action(state=state, dqn=self.dqn)
            next_state, reward = self.take_action(action.item())

            # Add experience to memory
            # State, action, reward, and next state are all tensors here
            self.memory.add_to_memory(Experience(state, action, reward, next_state))
            state = next_state

            # Learn on random batch from replay memory
            self.learn()

        print(f"Finished the episode after {num_steps} steps")



    def learn(self):
        try:
            replay_states, replay_actions, replay_rewards, replay_next_states = self.memory.sample(self.replay_batch_size)
            current_q_values = self.dqn.get_current_q_values(replay_states, replay_actions)
            next_q_values = self.dqn.get_next_q_values(replay_next_states)
            target_q_values = replay_rewards + (self.gamma * next_q_values)

            print(f"Target q values dims: {target_q_values.size()}")
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



        except Exception as e:
            # This could occur if not enough experiences in replay buffer yet
            print(f"Error in learning: {e}")
