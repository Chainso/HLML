import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import ACNetwork

class A2C(ACNetwork):
    """
    A neural network using the advantage actor-critic algorithm
    """
    def __init__(self, env, device, ent_coeff, vf_coeff, policy, lr, optimizer,
                 max_grad_norm=None):
        """
        Constructs the A2C model for the given environment

        env : The environment to run the model in
        device : The device to run the model on, either "cpu" for cpu or
                 "cuda" for gpu
        policy : The actor-critic policy network to train
        lr : The learning rate the optimizer
        optimizer : The optimizer for the A2C network to use
        ent_coeff : The coefficient of the entropy
        vf_coeff : The coefficient of the value loss
        max_grad_norm : The maximum value to clip the normalized gradients in
        """
        ACNetwork.__init__(self, env, device, ent_coeff, vf_coeff,
                           max_grad_norm)

        self.model = policy(*env.state_space(), env.action_space())

        self.optimizer = optimizer(self.model.parameters(), lr = lr)

    def train_batch(self, states, actions, returns):
        """
        Trains the network for a batch of (state, action, reward, next state)
        observations with the important sampling weights

        states : The observed states
        actions : The actions the network took in the states
        returns : The discounted returns for taking those actions in those
                  states
        """
        states = self.FloatTensor(states)
        actions = self.LongTensor(actions)
        returns = self.FloatTensor(returns)
