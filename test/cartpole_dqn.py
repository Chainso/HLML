import torch.nn as nn

from HLML.PyTorch.utils.classes import AMU

class Net(nn.Module):
    def __init__(self, state_space, act_space, hidden_size,
                 num_blocks, block_size, num_amus, sequence_length):
        nn.Module.__init__(self)

        self.sequence_length = sequence_length

        hiddens = [self._amu_block(hidden_size, block_size, num_blocks,
                                   hidden_size, 0.1) for _ in range(num_amus
                                                                    - 2)]

        self.lina = nn.Sequential(
            self._amu_block(state_space[0] // sequence_length, block_size,
                            num_blocks, hidden_size, 0.1),
            *hiddens,
            AMU(hidden_size, block_size, num_blocks, act_space, 0.1)
            )

        self.lin = remove_sequential(self)

    def _amu_block(self, input_units, block_units, num_blocks, output_units,
                   attn_dropout=0.1):
        return nn.Sequential(
            AMU(input_units, block_units, num_blocks, output_units,
                attn_dropout),
            nn.ReLU()
            )

    def reset_hidden(self):
        for module in self.modules():
            if(type(module) == AMU):
                module.reset_memory()

    def forward(self, inp):
        output = inp.view(inp.shape[0], self.sequence_length, -1)
        for block in self.lin:
            output = block(output)

            if(type(output) == tuple):
                output = output[0]

        output = output[:, -1, :]

        return output

def remove_sequential(module):
    all_layers = []
    for layer in module.children():
        if type(layer) == nn.Sequential: # if sequential layer, apply recursively to layers in sequential layer
            all_layers += remove_sequential(layer)
        else: # if leaf node, add it to list
            all_layers.append(layer)

    return all_layers

class Net2(nn.Module):
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
        a = self.lin(inp)
        print(a)
    
if(__name__ == "__main__"):
    import torch

    from torch.optim import Adam

    from HLML.PyTorch.RL.algos.dqn import DQN
    from HLML.envs.Gym import GymEnv, StackedGymEnv
    from HLML.utils.replay_memory import PERMemory
    from HLML.PyTorch.RL.agents.q_agent import DQNAgent

    # DQN Parameters
    stacked_frames = 4
    env = StackedGymEnv(1000, "CartPole-v1", False, stacked_frames)
    device = "cpu"
    save_path = "./DQN Models/dqn_retest.torch"
    save_interval = 1000
    policy = Net(env.state_space(), env.action_space(), 16, 3, 8, 2,
                 stacked_frames)
    #policy = Net2(env.state_space(), env.action_space(), 24, 2)
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
