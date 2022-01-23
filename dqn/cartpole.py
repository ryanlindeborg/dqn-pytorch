import torch
import torch.optim as optim

from epsilon_greedy_strategy import EpsilonGreedyStrategy
from env_manager import EnvManager
from dqn_agent import DQNAgent
from replay_memory import ReplayMemory
from dqn import DQN

OPENAI_ENV_CARTPOLE = "CartPole-v0"

def run_dqn_on_cartpole():
    #########################
    #### Hyperparameters ####
    #########################
    # Gamma is reward discount factor for each time step
    gamma = 0.999
    epsilon_start = 1
    epsilon_end = 0.01
    # epsilon_decay = 0.995
    # epsilon_decay = 0.9995
    epsilon_decay = 0.99995
    # epsilon_decay = 0.999995
    memory_size = 1000000
    lr = 0.001
    num_episodes = 100000
    num_eval_episodes = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Input and output dimensions for cartpole
    input_dims = 4
    output_dims = 2

    memory = ReplayMemory(
        memory_capacity=memory_size
    )
    dqn = DQN(
        input_dims=input_dims,
        output_dims=output_dims
    )
    target_dqn = DQN(
        input_dims=input_dims,
        output_dims=output_dims
    )
    optimizer = optim.Adam(params=dqn.parameters(), lr=lr)
    env_manager = EnvManager(
        env_name=OPENAI_ENV_CARTPOLE,
        device=device,
        memory=memory,
        dqn=dqn,
        target_dqn=target_dqn,
        optimizer=optimizer,
        gamma=gamma
    )
    exploration_strategy = EpsilonGreedyStrategy(
        start_exploration_rate=epsilon_start,
        min_exploration_rate=epsilon_end,
        decay_rate=epsilon_decay
    )
    dqn_agent = DQNAgent(
        exploration_strategy=exploration_strategy,
        num_actions=env_manager.get_num_actions_available(),
        device=device)
    env_manager.agent = dqn_agent

    # Run training loop for num_episodes
    for episode in range(num_episodes):
        print(f"Episode num: {episode}")
        if episode % 1000 == 0:
            print(f"Progress: On episode {episode}")
        env_manager.run_episode()
    env_manager.close_env()

    # Evaluate model
    env_manager.evaluate_model(num_eval_episodes)


if __name__ == "__main__":
    run_dqn_on_cartpole()