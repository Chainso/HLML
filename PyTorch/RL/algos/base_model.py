import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from tensorboardX import SummaryWriter

class Model(ABC, nn.Module):
    """
    An abstract RL model
    """
    def __init__(self, env, device):
        """
        Creates an abstract model

        env : The environment to run the model in
        device : The device to run the model on, either "cpu" for cpu or
                 "cuda" for gpu
        """
        nn.Module.__init__(self)

        self.env = env
        self.device = torch.device(device)

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

    def get_device(self):
        """
        Returns the device the model is being run on
        """
        return self.device

    def stop_training(self):
        """
        Stops training the model
        """
        self.started_training = False

    @abstractmethod
    def start_training(self, rollouts):
        """
        Starts training the network
        """
        pass

    @abstractmethod
    def step(self, observation):
        """
        Get the model output for a single observation of gameplay

        observation : A single observation from the environment
        """
        pass

    @abstractmethod
    def save(self, save_path):
        """
        Creates a save checkpoint of the model at the save path

        save_path : The path to save the checkpoint
        """
        pass

    @abstractmethod
    def load(self, load_path):
        """
        Loads the save checkpoint at the given load path

        load_path : The path of the checkpoint to load
        """
        pass

class ACNetwork(Model):
    """
    A neural network using the actor-critic model
    """
    def __init__(self, env, device, ent_coeff, vf_coeff, max_grad_norm=None):
        """
        Constructs an actor-critic network for the given environment

        env : The environment to run the model in
        device : The device to run the model on, either "cpu" for cpu or
                 "cuda" for gpu
        ent_coeff : The coefficient of the entropy
        vf_coeff : The coefficient of the value loss
        max_grad_norm : The maximum value to clip the normalized gradients in
        """
        Model.__init__(self, env, device)

        self.ent_coeff = ent_coeff
        self.vf_coeff = vf_coeff
        self.max_grad_norm = max_grad_norm

    def step(self, obs, greedy=True):
        """
        Get the model output for a single observation of gameplay

        obs : A single observation from the environment
        greedy : If true, the action will always be the action with the highest
                 advantages, otherwise, will be stochastic with the advantages
                 as weights

        Returns an action
        """
        adv, value = self.model(obs)
        adv = torch.exp(adv)

        if(greedy):
            adv = adv.argmax(1)
        else:
            adv = adv.multinomial(1)

        return adv.item(), value.item()
