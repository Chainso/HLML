import torch.nn as nn

class Net(nn.Module):
    def __init__(self, state_space, act_space, hidden_size, num_hidden):
        nn.Module.__init__(self)

        hiddens = [self._lin_block(hidden_size, hidden_size) for _
                   in range(num_hidden)]

        self.lin = nn.Sequential(
            self._lin_block(*state_space, hidden_size),
            *hiddens,
            self._lin_block(hidden_size, act_space)
            )

    def _lin_block(self, num_in, num_out):
        return nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.ReLU()
            )

    def forward(self, inp):
        return self.lin(inp)
    
if(__name__ == "__main__"):
    import torch

    from torch.optim import Adam

    from PyTorch.RL.algos.dqn import DQN
    from envs.Gym import GymEnv
    from utils.replay_memory import PERMemory
    from PyTorch.RL.agents.q_agent import DQNAgent

    # DQN Parameters
    env = GymEnv(1000, "CartPole-v1", False)
    device = "cpu"
    save_path = "./DQN Models/dqn_retest.torch"
    save_interval = 1000
    policy = Net(env.state_space(), env.action_space(), 24, 2)
    decay = 0.99
    target_update_interval = 25
    optimizer = Adam
    optimizer_params = {"lr" : 1e-3}
    dqn_params = (env, device, save_path, save_interval, policy, decay,
                  target_update_interval, optimizer, optimizer_params)
    dqn = DQN(*dqn_params).to(torch.device(device))
    #dqn.load(save_path)
    #dqn.eval()
    dqn.train()
    dqn.share_memory()

    # Prioritized Experience Replay Parameters
    capacity = 100000
    alpha = 0.7
    beta = 0.4
    beta_increment = 1e-3
    epsilon = 1e-3
    per_params = (capacity, alpha, beta, beta_increment, epsilon)
    per = PERMemory(*per_params)

    # Q-Agent Parameters
    n_steps = 200
    q_agent_params = (env, dqn, per, decay, n_steps)
    q_agent = DQNAgent(*q_agent_params)

    # Model Training Parameters
    batch_size = 32
    start_size = 128
    model_training_params = (per, batch_size, start_size)

    # Agent Training Parameters
    episodes = 1000
    agent_training_params = (episodes, batch_size, start_size)
    
    # Training
    q_agent.train(*agent_training_params)
