from PyTorch.RL.algos.dqn import DQN, QPolicy
from envs.Gym import GymEnv
from torch.optim import Adam

if(__name__ == "__main__"):
    env = GymEnv(1000, "CartPole-v1", True)
    device = "cpu"
    policy = QPolicy(env.state_space, env.action_space(), 16)
    decay = 0.99
    optimizer = Adam
    optimizer_params = {"lr" : 1e-3}
    dqn_params = (env, device, policy, decay, optimizer, optimizer_params)
    dqn = DQN(*dqn_params)

    