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
        self.device = device

        self.started_training = False
        self.steps_done = 0
        self.writer = None

        if(self.device == "cuda"):
            self.LongTensor = torch.cuda.LongTensor
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.LongTensor = torch.LongTensor
            self.FloatTensor = torch.FloatTensor

    def create_summary(self, logs_path, writer=None):
        """
        Creates a tensorboard summary of the loss of the network

        logs_path : The path to write the tensorboard logs to
        writer (optional) : If given, will ignore the logs path given and use
                            the summary writer given
        """
        if(writer is not None):
            self.writer = writer
        else:
            self.writer = SummaryWriter(logs_path)

    def stop_training(self):
        """
        Stops training the model
        """
        self.started_training = False

    def get_device(self):
        """
        Returns the device the model is being run on
        """
        return self.device

    @abstractmethod
    def step(self, state):
        """
        Get the model output for a single state of gameplay

        state : A single state from the environment

        Returns a tuple of (action, state value)
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


class QNetwork(Model):
    """
    A general Q Network
    """
    def __init__(self, env, device, policy, lr, decay, optimizer):
        """
        Constructs a Q Network for the given environment

        env : The environment to run the model in
        device : The device to run the model on, either "cpu" for cpu or
                 "cuda" for gpu
        policy : The Q policy network to train
        lr : The learning rate the optimizer
        decay : The gamma value for the Q-value decay
        optimizer : The optimizer for the Q network to use
        """
        Model.__init__(self, env, device)

        self.online = policy(*env.state_space(), env.action_space())
        self.target = policy(*env.state_space(), env.action_space())
        self.update_target()

        self.decay = decay
        self.optimizer = optimizer(self.online.parameters(), lr = lr)

    def start_training(self, replay_memory, batch_size, start_size):
        """
        Continually trains on the replay memory in a separate process

        replay_memory : The replay memory for the model to use
        batch_size : The batch size of the samples to train on
        start_size : The size of the replay_memory before starting training
        """
        self.started_training = True

        while(self.started_training):
            if(replay_memory.size() >= start_size):
                batch, idxs, is_weights = replay_memory.sample(batch_size)

                states, actions, rewards, next_states = zip(*batch)

                states = self.FloatTensor(states)
                actions = self.LongTensor(actions)
                rewards = self.FloatTensor(rewards)
                next_states = self.FloatTensor(next_states)
                is_weights = self.FloatTensor(is_weights)

                losses = self.train_batch(states, actions, rewards, next_states,
                                          is_weights).cpu().data.numpy()

                replay_memory.update_weights(idxs, losses)

    def update_target(self):
        """
        Updates the target network with the weights of the online network
        """
        self.target.load_state_dict(self.online.state_dict())

    def save(self, save_path):
        """
        Creates a save checkpoint of the model at the save path

        save_path : The path to save the checkpoint
        """
        save_dict = {"state_dict" : self.online.state_dict(),
                     "optimizer" : self.optimizer.state_dict(),
                     "steps_done" : self.steps_done}
        torch.save(save_dict, save_path)

    def load(self, load_path):
        """
        Loads the save checkpoint at the given load path

        load_path : The path of the checkpoint to load
        """
        save_state = torch.load(load_path)

        self.online.load_state_dict(save_state["state_dict"])
        self.target.load_state_dict(save_state["state_dict"])
        self.optimizer.load_state_dict(save_state["optimizer"])
        self.steps_done = save_state["steps_done"]

    @abstractmethod
    def train_batch(self, states, actions, returns, next_states, is_weights):
        """
        Trains the network for a batch of (state, action, reward, next state)
        observations with the important sampling weights

        states : The observed states
        actions : The actions the network took in the states
        rewards : The rewards for taking those actions in those states
        next_states : The resulting state after taking the action
        is_weights : The important sampling weights in prioritized experience
                     replay
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

    def step(self, state, greedy=True):
        """
        Get the model output for a single state of gameplay

        state : A single state from the environment
        greedy : If true, the action will always be the action with the highest
                 advantages, otherwise, will be stochastic with the advantages
                 as weights

        Returns an action
        """
        adv, value = self.model(state)
        adv = torch.exp(adv)

        if(greedy):
            adv = adv.argmax(1)
        else:
            adv = adv.multinomial(1)

        return adv.item(), value.item()

    #@abstractmethod
    def train_batch(self, states, actions, returns):
        """
        Trains the network for a batch of (state, action, reward, next state)
        observations with the important sampling weights

        states : The observed states
        actions : The actions the network took in the states
        returns : The discounted returns for taking those actions in those
                  states
        values : The value of the states
        """
        pass
