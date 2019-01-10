import numpy as np

from abc import ABC, abstractmethod
from tensorboardX import SummaryWriter

class Agent(ABC):
    """
    An agent that collects observations from its environment
    """
    def __init__(self, env, model, save_path=None, logs_path=None, writer=None):
        """
        Creates an agent to collect the observations

        env : The environment to collect observations from
        model : The model for the agent to play on
        save_path : The path to save the model to during training
        logs_path : The logs path to create the writer for if a summary writer
                    isn't given
        writer : The summary writer for the agent
        """
        self.env = env
        self.model = model
        self.save_path = save_path
        self.logs_path = logs_path
        self.writer = writer

        # If there isn't a given writer then make one with the given logs path
        if(self.writer is None and self.logs_path is not None):
            self.writer = SummaryWriter(self.logs_path)

    def _process_state(self, state):
        """
        Takes in a state and prepares it to be fed into the model

        state : The state to prepare

        Returns the prepared state
        """
        return self.model.FloatTensor(np.stack([state]))

    def _create_summary(self, logs_path, writer=None):
        """
        Creates a tensorboard summary of the agent rewards and a summary
        of the network losses

        logs_path : The path to write the tensorboard logs to
        writer (optional) : If given, will ignore the logs path given and use
                            the summary writer given
        """
        if(writer is not None):
            self.writer = writer
        else:
            self.writer = SummaryWriter(logs_path)

        self.model.create_summary("", self.writer)

    @abstractmethod
    def train(self, episodes, logs_path=None):
        """
        Starts the training for the network

        episodes : The number of episodes to train for
        logs_path : The path to save the tensorboard graphs during training
        """
        for episode in range(episodes):
            self.step(logs_path)

    @abstractmethod
    def step(self, logs_path=None):
        """
        Causes the agent to take 1 step in the environment

        logs_path : The path to save the tensorboard graphs during training
                    and playing
        """
        