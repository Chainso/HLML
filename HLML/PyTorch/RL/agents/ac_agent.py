import numpy as np

from .base_agent import Agent

from HLML.PyTorch.utils import normalize
from HLML.PyTorch.RL.utils import discount

class ACAgent(Agent):
    """
    An agent that collects (state, action, reward, advantage) observations
    from an environment
    """
    def __init__(self, env, ac_network, replay_memory, decay, n_steps):
        """
        Creates an agent to collect the (observation, action, reward, advantage)
        rollouts and train the actor critic network on them

        env : The environment to collect observations from
        ac_network : The actor critic model for the agent to play on
        replay_memory : The replay memory to store the observations in
        decay : The decay rate for the n step discount
        n_steps : The number of steps for the discounted returns
        """
        Agent.__init__(self, env, ac_network)

        self.replay_memory = replay_memory
        self.decay = decay
        self.n_steps = n_steps

    def train(self, episodes, batch_size, start_size, *training_args):
        """
        Starts the agent in the environment

        episodes : The number of episodes to train for
        batch_size : The size of each training batch
        start_size : The number of experiences to accumulate before starting
                     training
        training_args : Any additional training arguments for the actor critic
                        network
        """
        if(self.writer is not None):
            self.model.create_summary("Agent", self.writer)

        for episode in range(1, episodes + 1):
            done = False
            total_reward = 0

            state = self.env.reset()
            state = self._process_state(state)

            states = []
            actions = []
            rewards = []
            values = []
            dones = []
            errors = []

            while(not done):
                step = 0

                while(not done and step < self.n_steps):
                    action, value = self.model.step(state)
    
                    next_state, reward, done = self.env.step(action)
    
                    next_state = self._process_state(next_state)
                    total_reward += reward
    
                    #error = torch.abs(value - reward)
                    error = 0

                    states += state.cpu().numpy().tolist()
                    actions.append(action)
                    rewards.append(reward)
                    values.append(value)
                    dones.append(done)
                    errors.append(error)
    
                    state = next_state

                _, last_val = self.model.step(next_state)
                values += [last_val]

                advantages = (rewards + self.decay * (1 - np.array(dones))
                              * values[1:])
                advantages = discount(advantages, self.decay) - values[:-1]
                advantages = normalize(advantages, 1e-8)

                rewards = discount(rewards, self.decay)

                self.model.train_batch(states, actions, rewards,
                                            advantages, *training_args)

            print("Episode", str(episode) + ":", total_reward)

            if(self.writer is not None):
                self.writer.add_scalar("Train/Reward", total_reward,
                                       self.model.steps_done)
