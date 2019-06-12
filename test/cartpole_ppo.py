import torch.nn as nn

class Net(nn.Module):
    def __init__(self, state_space, act_space, hidden_size, num_hidden):
        nn.Module.__init__(self)

        hiddens = [self._lin_block(hidden_size, hidden_size) for _
                   in range(num_hidden)]

        self.lin = nn.Sequential(
            self._lin_block(*state_space, hidden_size),
            *hiddens
            )

        self.policy = nn.Linear(hidden_size, act_space)
        self.value = nn.Linear(hidden_size, 1)

    def _lin_block(self, num_in, num_out):
        return nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.ReLU()
            )

    def forward(self, inp):
        lin = self.lin(inp)

        policy = self.policy(lin)
        value = self.value(lin)

        return policy, value
    
if(__name__ == "__main__"):
    import torch

    from torch.optim import Adam

    from HLML.PyTorch.RL.algos.ppo import PPO
    from HLML.envs.Gym import GymEnv
    from HLML.utils.replay_memory import PERMemory
    from HLML.PyTorch.RL.agents.ac_agent import ACAgent

    # PPO Parameters
    env = GymEnv(1000, "CartPole-v1", False)
    device = "cuda"
    save_path = "./PPO Models/ppo_cartpole.torch"
    save_interval = 1000
    policy = Net(env.state_space(), env.action_space(), 24, 2)
    target_update_interval = 25
    ent_coeff = 1e-2
    vf_coeff = 0.5
    clip_range = 0.2
    optimizer = Adam
    optimizer_params = {"lr" : 1e-3}
    ppo_params = (env, device, save_path, save_interval, policy,
                  target_update_interval, ent_coeff, vf_coeff, clip_range,
                  optimizer, optimizer_params)
    ppo = PPO(*ppo_params).to(torch.device(device))
    #ppo.load(save_path)
    #ppo.eval()
    ppo.train()

    # Prioritized Experience Replay Parameters
    capacity = 100000
    alpha = 0.7
    beta = 0.4
    beta_increment = 1e-3
    epsilon = 1e-3
    per_params = (capacity, alpha, beta, beta_increment, epsilon)
    per = PERMemory(*per_params)

    # actor critic agent Parameters
    decay = 0.99
    n_steps = 200
    ac_agent_params = (env, ppo, per, decay, n_steps)
    ac_agent = ACAgent(*ac_agent_params)

    # Model Training Parameters
    batch_size = 32
    start_size = 128
    n_times_per_batch = 4
    model_training_params = (per, batch_size, start_size)

    # Agent Training Parameters
    episodes = 1000
    training_args = (n_times_per_batch,)
    agent_training_params = (episodes, batch_size, start_size, *training_args)
    
    # Training
    ac_agent.train(*agent_training_params)
