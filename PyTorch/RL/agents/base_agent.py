import numpy as np

from abc import ABC, abstractmethod
from tensorboardX import SummaryWriter

class Agent(ABC):
    """
    An agent that collects observations from its environment
    """
    def __init__(self, env, model, save_path=None):
        """
        Creates an agent to collect the observations

        env : The environment to collect observations from
        model : The model for the agent to play on
        save_path : The path to save the model to during training
        """
        self.env = env
        self.model = model
        self.save_path = save_path
        self.writer = None

    def _process_state(self, state):
        """
        Takes in a state and prepares it to be fed into the model

        state : The state to prepare

        Returns the prepared state
        """
        return self.model.FloatTensor(np.stack([state]))

    def create_summary(self, logs_path, writer=None):
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
    def train(self):
        pass
