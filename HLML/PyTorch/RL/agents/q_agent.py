import torch

from .base_agent import Agent
from PyTorch.RL.utils import discount, normalize

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

    def train(self, episodes, batch_size, start_size):
        """
        Starts the agent in the environment

        episodes : The number of episodes to train for
        batch_size : The size of each training batch
        start_size : The number of experiences to accumulate before starting
                     training
        """
        if(self.writer is not None):
            self.model.create_summary("Agent", self.writer)

        for episode in range(1, episodes + 1):
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
                error = torch.abs(reward + self.decay *
                                  self.model.online(next_state)[0, next_act] -
                                  q_value).item()

                states += state.cpu().numpy().tolist()
                actions.append(action)
                rewards.append(reward)
                next_states += next_state.cpu().numpy().tolist()
                errors.append(error)

                if(len(states) == self.n_steps):
                    rewards = discount(rewards, self.decay)

                    for s, a, r, n_s, err in zip(states, actions, rewards,
                                                 next_states, errors):
                        self.replay_memory.add([s, a, r, n_s], err)

                    states = []
                    actions = []
                    rewards = []
                    next_states = []
                    errors = []

                if(len(self.replay_memory) >= start_size):
                    sample = self.replay_memory.sample(batch_size)
                    rollouts, indices, is_weights = sample
    
                    loss, new_errors = self.model.train_batch(rollouts,
                                                              is_weights)
                    self.replay_memory.update_priorities(indices, new_errors)

                state = next_state

            if(len(states) > 0):
                rewards = discount(rewards, self.decay)
                rewards = normalize(rewards)
                rewards = torch.from_numpy(rewards.copy()).float()

                for s, a, r, n_s, err in zip(states, actions, rewards,
                                             next_states, errors):
                    self.replay_memory.add((s, a, r, n_s), err)

            print("Episode", str(episode) + ":", tot_reward)

            if(self.writer is not None):
                self.writer.add_scalar("Train/Reward", tot_reward,
                                       self.model.steps_done)
