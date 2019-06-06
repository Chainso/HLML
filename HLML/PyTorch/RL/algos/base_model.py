import torch
import torch.nn as nn

from abc import abstractmethod
from tensorboardX import SummaryWriter
from torch.distributions import Categorical

from PyTorch.base_model import Model

class RLModel(Model):
    """
    An abstract RL model
    """
    def __init__(self, env, device, save_path, save_interval):
        """
        Creates an abstract model

        env : The environment to run the model in
        device : The device to run the model on, either "cpu" for cpu or
                 "cuda" for gpu
        save_path : The path to save the model to
        save_interval : The number of steps in between model saves
        """
        Model.__init__(self, device)

        self.env = env
        self.save_path = save_path
        self.save_interval = save_interval

        self.started_training = False
        self.steps_done = 0
        self.writer = None

    def create_summary(self, logs_path=None, writer=None):
        """
        Creates a tensorboard summary of the loss of the network

        logs_path : The path to write the tensorboard logs to
        writer (optional) : If given, will ignore the logs path given and use
                            the summary writer given
        """
        assert not (logs_path == writer == None)

        if(writer is not None):
            self.writer = writer
        else:
            self.writer = SummaryWriter(logs_path)

    def stop_training(self):
        """
        Stops training the model
        """
        self.started_training = False

    def routine_save(self):
        """
        Checks to see if the number of steps done is a multiple of the save
        interval and will save the model if it is
        """
        if(self.steps_done % self.save_interval == 0 and self.training):
            self.save(self.save_path)

    @abstractmethod
    def start_training(self, rollouts):
        """
        Starts training the network
        """
        pass

    @abstractmethod
    def train_batch(self, rollouts):
        """
        Trains the network for a batch of rollouts

        rollouts : The rollouts of training data for the network
        """
        pass

    @abstractmethod
    def step(self, observation):
        """
        Get the model output for a single observation of gameplay

        observation : A single observation from the environment
        """
        pass

class ACNetwork(RLModel):
    """
    A neural network using the actor-critic model
    """
    def __init__(self, env, device, save_path, save_interval, ent_coeff,
                 vf_coeff, max_grad_norm=None):
        """
        Constructs an actor-critic network for the given environment

        env : The environment to run the model in
        device : The device to run the model on, either "cpu" for cpu or
                 "cuda" for gpu
        save_path : The path to save the model to
        save_interval : The number of steps in between each save of the model
        ent_coeff : The coefficient of the entropy
        vf_coeff : The coefficient of the value loss
        max_grad_norm : The maximum value to clip the normalized gradients in
        """
        RLModel.__init__(self, env, device, save_path, save_interval)

        self.ent_coeff = ent_coeff
        self.vf_coeff = vf_coeff
        self.max_grad_norm = max_grad_norm

    def step(self, obs, greedy=False):
        """
        Get the model output for a single observation of gameplay

        obs : A single observation from the environment
        greedy : If true, the action will always be the action with the highest
                 advantages, otherwise, will be stochastic with the advantages
                 as weights

        Returns an action
        """
        with torch.no_grad():
            adv, value = self.model(obs)
            adv = nn.Softmax(-1)(adv)
            adv = Categorical(adv)
    
            action = adv.sample()

            self.steps_done += 1
            self.routine_save()
    
            return action.item(), value.item()
