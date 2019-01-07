from numpy import exp
from numpy.random import rand, choice

from .base_agent import Agent

from PyTorch.utils import normalize
from PyTorch.RL.utils import discount

class ACAgent(Agent):
    """
    An agent that collects (state, action, reward, advantage) observations
    from an environment
    """
    def __init__(self, env, replay_memory, ac_network, decay, save_path=None):
        """
        Creates an agent to collect the (state, action, reward, next state)
        pairs and store them in the replay memory

        env : The environment to collect observations from
        replay_memory : The replay memory to store the observations in
        ac_network : The actor-critic model for the agent to play on
        decay : The decay rate for the n step discount
        save_path : The path to save the model to during training
        """
        Agent.__init__(self, env, ac_network, save_path)

        self.replay_memory = replay_memory
        self.decay = decay

        self.epsilon_start = 0.9
        self.epsilon_end = 0.1
        self.epsilon_decay = 200

    def train(self, episodes, logs_path=None):
        """
        Starts the agent in the environment

        episodes : The number of episodes to train for
        logs_path : The path to save the tensorboard graphs during training
                    and playing
        """
        if(logs_path is not None):
            self.create_summary(logs_path)

        for episode in range(episodes):
            done = False
            total_reward = 0

            state = self.env.reset()
            state = self._process_state(state)

            states = []
            actions = []
            rewards = []
            values = []
            errors = []

            while(not done):
                epsilon = (self.epsilon_end + (self.epsilon_start -
                                               self.epsilon_end) *
                           exp(-1. * self.model.steps_done / self.epsilon_decay))

                action, value = self.model.step(state)

                if(rand() < epsilon):
                    action = choice(self.env.action_space())

                next_state, reward, done = self.env.step(action)

                next_state = self._process_state(next_state)
                total_reward += reward

                error = 0.5 * (value - reward) ** 2

                states += state.cpu().numpy().tolist()
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                errors.append(error)

                state = next_state
                self.model.steps_done += 1

            if(len(states) > 0):
                returns = discount(rewards, self.decay)
                advantages = returns - values
                advantages = normalize(advantages, 1e-8)

                for s, a, r, adv, err in zip(states, actions, returns,
                                             advantages, errors):
                    self.replay_memory.add((s, a, r, adv), err)

            if(self.save_path is not None and (episode + 1) % 25 == 0):
                self.model.save(self.save_path)
                self.model.update_target()

            print(total_reward)

            if(self.writer is not None):
                self.writer.add_scalar("Train/Reward", total_reward,
                                       self.model.steps_done)
