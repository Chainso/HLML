import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base_model import QNetwork
from HLML.PyTorch.utils import to_device

class DQN(QNetwork):
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
        QNetwork.__init__(self, env, device, policy, lr, decay, optimizer)

        self.EPS_START = 0.9
        self.EPS_END = 0.01
        self.EPS_DECAY = 1000

    def step(self, state):
        with torch.no_grad():
            eps_threshold = (self.EPS_END + (self.EPS_START - self.EPS_END) *
                             np.exp(-1. * self.steps_done / self.EPS_DECAY))

            if(np.random.rand(1) < eps_threshold):
                action = torch.from_numpy(np.array([np.random.choice(np.arange(self.env.action_space()))])).long().item()
            else:
                action = self.online(state).argmax(1).item()

            q_value = self.online(state)[0, action]

            return action, q_value

    def train_batch(self, states, actions, rewards, next_states, is_weights):
        state_vals = to_device(self.online, self.device)(states).gather(1, actions.unsqueeze(1)).view(-1,)

        next_state_act = to_device(self.online, self.device)(next_states).detach().argmax(1)

        next_state_vals = to_device(self.target, self.device)(next_states).detach()
        next_state_vals = next_state_vals.gather(1, next_state_act.unsqueeze(1)).view(-1,)

        target_q = (rewards + self.decay * next_state_vals)

        losses = 0.5 * ((state_vals - target_q) ** 2)

        loss = (is_weights * losses).mean()

        if(self.writer is not None):
            self.writer.add_scalar("Train/Loss", loss, self.steps_done)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return losses


class QPolicy(nn.Module):
    """
    A policy function of a QNetwork
    """
    def __init__(self, state_space, act_space):
        """
        Creates the policy network

        state_space : The number of input units for the state
        act_space : The number of output units for the action
        """
        nn.Module.__init__(self)

        self.layer1 = nn.Linear(state_space, 16)

        self.advantage = nn.Linear(16, act_space)
        self.value = nn.Linear(16, 1)

    def forward(self, inp):
        inp = F.relu(self.layer1(inp))

        advantage = self.advantage(inp)
        value = self.value(inp)

        return value + advantage - advantage.mean(1, keepdim=True)
