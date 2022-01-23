import gym
import torch
import torch.nn.functional as F

from experience import Experience

class EnvManager():
    def __init__(self, env_name, device, memory, dqn, target_dqn, optimizer, gamma=1, agent=None):
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
        self.target_dqn = target_dqn
        self.optimizer = optimizer
        self.gamma = gamma
        self.agent = agent
        self.device = device
        # Keep track of backpropagation updates made, so that can update target Q network at lower frequency
        self.num_network_param_updates = 0
        # Number of network updates before target network is updated
        self.target_network_update_frequency = 300

    def get_num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.done = done
        return torch.tensor(next_state, dtype=torch.float32, device=self.device), torch.tensor([reward], device=self.device)

    def close_env(self):
        self.env.close()

    def run_episode(self):
        num_steps = 0
        score = 0
        # TODO: Can play with changing state to be from rendered screen instead of returned state from environment
        state = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        while not self.done and num_steps < 1000:
            num_steps += 1
            # Select and take action
            action = self.agent.get_action(state=state, dqn=self.dqn)
            next_state, reward = self.take_action(action.item())
            score += reward.item()
            # print(f"Num steps: {num_steps}. Reward: {reward.item()}, current score: {score}")

            # Add experience to memory
            # State, action, reward, and next state are all tensors here
            self.memory.add_to_memory(Experience(state=state, action=action, reward=reward, next_state=next_state))
            state = next_state

            # Learn on random batch from replay memory
            self.learn()

        print(f"Epsilon: {self.agent.get_epsilon()}")
        print(f"Finished the episode after {num_steps} steps. Reward: {score}")
        self.done = False

    # CartPole-v0 defines solving the env as getting average reward of 195 over 100 consecutive trials
    def evaluate_model(self, num_eval_episodes):
        self.dqn.eval()
        for _ in range(num_eval_episodes):
            num_steps = 0
            score = 0
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            while not self.done and num_steps < 1000:
                num_steps += 1
                # Select and take action
                action = self.agent.get_action(state=state, dqn=self.dqn, is_training=False)
                next_state, reward = self.take_action(action.item())
                score += reward.item()
                state = next_state
            self.done = False

        average_reward = score / num_eval_episodes
        if average_reward > 195:
            print(f"SUCCESS! You have solved Cartpole, with an average score of {average_reward} over {num_eval_episodes} episodes")
        else:
            print(f"Darn...you did not solve Cartpole, with an average score of {average_reward} over {num_eval_episodes} episodes")

    def learn(self):
        try:
            self.dqn.train()
            replay_states, replay_actions, replay_rewards, replay_next_states = self.memory.sample(self.replay_batch_size)
            current_q_values = self.dqn.get_current_q_values(replay_states, replay_actions)
            next_q_values = self.target_dqn.get_next_q_values(replay_next_states)
            target_q_values = replay_rewards + (self.gamma * next_q_values)

            # print(f"Target q values dims: {target_q_values.size()}")
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            for param in self.dqn.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            self.num_network_param_updates += 1

            # Periodically update the target network by Q network to target Q network
            if self.num_network_param_updates % self.target_network_update_frequency == 0:
                # print("*************Made target network update")
                # print("*************Made target network update")
                # print("*************Made target network update")
                # print("*************Made target network update")
                # print("*************Made target network update")
                self.target_dqn.load_state_dict(self.dqn.state_dict())
        except Exception as e:
            # This could occur if not enough experiences in replay buffer yet
            print(f"Error in learning: {e}")
