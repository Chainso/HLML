import torch
import torch.nn as nn
import numpy as np

from torch.distributions import Categorical
from copy import deepcopy

from .base_model import Model

class QNetwork(Model):
    """
    A general Q Network
    """
    def __init__(self, env, device, save_path, save_interval, policy, decay,
                 target_update_interval, optimizer, optimizer_params):
        """
        Constructs a Q Network for the given environment

        env : The environment to run the model in
        device : The device to run the model on, either "cpu" for cpu or
                 "cuda" for gpu
        save_path : The path to save the model to
        save_interval : The number of steps in between model saves
        policy : The Q policy network to train
        decay : The gamma value for the Q-value decay
        target_update_interval : The number of steps in between each target
                                 update
        optimizer : The optimizer for the Q network to use
        optimizer_params : The parameters for the optimizer (including learning
                           rate)
        """
        Model.__init__(self, env, device, save_path, save_interval)

        self.online = policy
        self.target = deepcopy(policy)
        self.target_update_interval = target_update_interval
        self.update_target()

        self.decay = decay

        self.optimizer = optimizer(self.online.parameters(), **optimizer_params)

    def start_training(self, replay_memory, batch_size, start_size):
        """
        Continually trains on the replay memory in a separate process

        replay_memory : The replay memory for the model to use
        batch_size : The batch size of the samples to train on
        start_size : The size of the replay_memory before starting training
        """
        self.started_training = True

        while(self.started_training):
            if(len(replay_memory) >= start_size):
                sample = replay_memory.sample(batch_size)
                rollouts, args = sample[0], sample[1:]

                self.train_batch(rollouts, *args)

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

class DQN(QNetwork):
    def __init__(self, env, device, save_path, save_interval, policy, decay,
                 target_update_interval, optimizer, optimizer_params):
        """
        Constructs a Q Network for the given environment

        env : The environment to run the model in
        device : The device to run the model on, either "cpu" for cpu or
                 "cuda" for gpu
        save_path : The path to save the model to
        save_interval : The number of steps in between model saves
        policy : The Q policy network to train
        decay : The gamma value for the Q-value decay
        target_update_interval : The number of steps in between each target
                                 update
        optimizer : The optimizer for the Q network to use
        optimizer_params : The parameters for the optimizer (including learning
                           rate)
        """
        QNetwork.__init__(self, env, device, save_path, save_interval, policy,
                          decay, target_update_interval, optimizer,
                          optimizer_params)

    def step(self, state):
        """
        Get the model output for a single observation of gameplay

        observation : A single observation from the environment

        Returns a tuple of (action, Q-value)
        """
        with torch.no_grad():
            action = nn.Softmax(-1)(self.online(state))
            action = Categorical(action)
            action = action.sample().item()

            q_value = self.online(state)[0, action]
            self.steps_done += 1

            if(self.steps_done % self.target_update_interval == 0):
                self.update_target()

            self.routine_save()

            return action, q_value

    def train_batch(self, rollouts, is_weights):
        """
        Trains the network for a batch of rollouts, returns the loss and the
        new errors for the experiences

        rollouts : The rollouts of (observations, actions, rewards, next
                   observations) for the network
        is_weights : The importance sampling weights for the experiences
        """
        tens_rollouts = [torch.from_numpy(arr).to(self.device) for arr in rollouts]
        obs, actions, rewards, next_obs = tens_rollouts

        is_weights = torch.FloatTensor(is_weights).to(self.device)

        q_vals = self.online(obs)
        q_vals = q_vals.gather(1, actions.unsqueeze(1)).view(-1,)

        next_acts = self.online(next_obs).detach().argmax(1)

        next_q_vals = self.target(next_obs).detach()
        next_q_vals = next_q_vals.gather(1, next_acts.unsqueeze(1)).view(-1,)

        target_q = rewards + self.decay * next_q_vals

        loss_func = nn.MSELoss(reduction="none")
        losses = loss_func(q_vals, target_q)
        loss = (is_weights * losses).mean()

        if(self.writer is not None):
            self.writer.add_scalar("Train/Loss", loss, self.steps_done)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        online_next_q = self.online(next_obs).gather(1, next_acts.unsqueeze(1)).view(-1,)
        errors = torch.abs(rewards + self.decay * online_next_q - q_vals).detach().cpu().numpy()

        return loss, errors

class QPolicy(nn.Module):
    """
    A policy function of a QNetwork
    """
    def __init__(self, state_space, act_space, hidden_size = 16):
        """
        Creates the policy network

        state_space : The number of input units for the state
        act_space : The number of output units for the action
        """
        nn.Module.__init__(self)

        self.linear = nn.Sequential(
            nn.Linear(*state_space, hidden_size),
            nn.ReLU()
            )

        self.advantage = nn.Linear(hidden_size, act_space)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, inp):
        """
        Runs the Q-network on the given input

        inp : The input to run the Q-network on
        """
        inp = self.linear(inp)

        advantage = self.advantage(inp)
        value = self.value(inp)

        q_vals =  value + advantage - advantage.mean(1, keepdim=True)

        return q_vals
