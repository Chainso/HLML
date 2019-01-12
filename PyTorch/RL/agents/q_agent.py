import torch

from .base_agent import Agent
from PyTorch.RL.utils import discount

class DQNAgent(Agent):
    """
    An agent that collects (observation, action, reward, next observation)
    rollouts from an environment
    """
    def __init__(self, env, dqn, replay_memory, decay, n_steps):
        """
        Creates an agent to collect the (observation, action, reward, next
        observation) rollouts and store them in the replay memory

        env : The environment to collect observations from
        dqn : The DQN model for the agent to play on
        replay_memory : The replay memory to store the observations in
        decay : The decay rate for the n step discount
        n_steps : The number of steps for the discounted reward
        """
        Agent.__init__(self, env, dqn)

        self.replay_memory = replay_memory
        self.decay = decay
        self.n_steps = n_steps

    def train(self, episodes):
        """
        Starts the agent in the environment

        episodes : The number of episodes to train for
        """
        if(self.writer is not None):
            self.model.create_summary("", self.writer)

        for episode in range(episodes):
            done = False

            tot_reward = 0

            state = self.env.reset()
            state = self._process_state(state)

            states = []
            actions = []
            rewards = []
            next_states = []
            errors = []

            while(not done):
                action, q_value = self.model.step(state)

                next_state, reward, done = self.env.step(action)

                next_state = self._process_state(next_state)
                tot_reward += reward

                next_act = self.model.target(next_state).argmax(1)
                error = 0.5 * (reward + self.decay *
                               self.model.online(next_state).detach().numpy()[0, next_act] -
                               q_value.detach().numpy() ** 2)

                states += state.numpy().tolist()
                actions.append(action)
                rewards.append(reward)
                next_states += next_state.numpy().tolist()
                errors.append(error)

                if(len(states) == self.n_steps):
                    rewards = discount(rewards, self.decay)
                    rewards = torch.from_numpy(rewards.copy()).float()

                    for s, a, r, n_s, err in zip(states, actions, rewards,
                                                 next_states, errors):
                        self.replay_memory.add((s, a, r, n_s), err)

                    states = []
                    actions = []
                    rewards = []
                    next_states = []
                    errors = []

                state = next_state

                self.model.steps_done += 1

            if(len(states) > 0):
                rewards = discount(rewards, self.decay)
                rewards = torch.from_numpy(rewards.copy()).float()

                for s, a, r, n_s, err in zip(states, actions, rewards,
                                             next_states, errors):
                    self.replay_memory.add((s, a, r, n_s), err)

            print(tot_reward)

            if(self.writer is not None):
                self.writer.add_scalar("Train/Reward", tot_reward,
                                       self.model.steps_done)
