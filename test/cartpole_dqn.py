if(__name__ == "__main__"):
    from torch.multiprocessing import Process
    from torch.optim import Adam

    from PyTorch.RL.algos.dqn import DQN, QPolicy
    from envs.Gym import GymEnv
    from utils.replay_memory import PERMemory
    from PyTorch.RL.agents.q_agent import DQNAgent

    # DQN Parameters
    env = GymEnv(1000, "CartPole-v1", False)
    device = "cpu"
    save_path = "./Cartpole Models/dqn.torch"
    save_interval = 50
    policy = QPolicy(env.state_space(), env.action_space(), 16)
    decay = 0.99
    target_update_interval = 25
    optimizer = Adam
    optimizer_params = {"lr" : 1e-3}
    dqn_params = (env, device, save_path, save_interval, policy, decay,
                  target_update_interval, optimizer, optimizer_params)
    dqn = DQN(*dqn_params)

    # Prioritized Experience Replay Parameters
    capacity = 50000
    alpha = 0.7
    beta = 0.4
    beta_increment = 1e-3
    epsilon = 1e-3
    per_params = (capacity, alpha, beta, beta_increment, epsilon)
    per = PERMemory(*per_params)

    # Q-Agent Parameters
    n_steps = 32
    q_agent_params = (env, dqn, per, decay, n_steps)
    q_agent = DQNAgent(*q_agent_params)

    # Model Training Parameters
    batch_size = 128
    start_size = 128
    model_training_params = (per, batch_size, start_size)

    # Agent Training Parameters
    episodes = 1000
    agent_training_params = (episodes, batch_size, start_size)

    # Training Processes
    dqn.share_memory()

    model_proc = Process(target=dqn.start_training, args=model_training_params)
    agent_proc = Process(target=q_agent.train, args=agent_training_params)

   # model_proc.start()
    agent_proc.start()

    #model_proc.join()
    agent_proc.join()
