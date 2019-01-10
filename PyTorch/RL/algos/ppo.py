import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import ACNetwork

class PPO(ACNetwork):
    """
    A neural network using the proximal policy optimization algorithm
    """
    def __init__(self, env, device, ent_coeff, vf_coeff, policy, lr, optimizer,
                 clip_range, max_grad_norm=None):
        """
        Constructs an PPO network for the given environment

        env : The environment to run the model in
        device : The device to run the model on, either "cpu" for cpu or
                 "cuda" for gpu
        ent_coeff : The coefficient of the entropy
        vf_coeff : The coefficient of the value loss
        policy : The actor-critic policy network to train
        lr : The learning rate the optimizer
        optimizer : The optimizer for the PPO network to use
        clip_range : The range to clip the surrogate objective in
                     (1 - clip_range, 1 + clip_range)
        max_grad_norm : The maximum value to clip the normalized gradients in
        """
        ACNetwork.__init__(self, env, device, ent_coeff, vf_coeff,
                           max_grad_norm)

        self.model = policy(*env.state_space(), env.action_space())
        self.old_model = policy(*env.state_space(), env.action_space())

        self.clip_range = clip_range

        self.optimizer = optimizer(self.model.parameters(), lr = lr)

    def update_target(self):
        """
        Updates the old network with the weights of the actor network
        """
        self.old_model.load_state_dict(self.model.state_dict())

    def start_training(self, replay_memory, batch_size, start_size,
                       n_times_batch):
        """
        Continually trains on the replay memory in a separate process

        replay_memory : The replay memory for the model to use
        batch_size : The batch size of the samples to train on
        start_size : The size of the replay_memory before starting training
        n_times_batch : The number of times to train per batch
        """
        self.started_training = True

        while(self.started_training):
            if(replay_memory.size() >= start_size):
                batch, idxs, _ = replay_memory.sample(batch_size)

                states, actions, rewards, advantages = zip(*batch)

                states = self.FloatTensor(states)
                actions = self.LongTensor(actions)
                rewards = self.FloatTensor(rewards)
                advantages = self.FloatTensor(advantages)

                losses = self.train_batch(states, actions, rewards, advantages,
                                          n_times_batch).cpu().data.numpy()

                replay_memory.update_weights(idxs, losses)

    def train_batch(self, states, actions, rewards, advantages, n_times_batch):
        """
        Trains the network for a batch of (state, action, reward, advantage)
        observations

        states : The observed states
        actions : The actions the network took in the states
        rewards : The rewards for taking those actions in those states
        advantage : The advantage of taking those actions in those states
        n_times_batch : The number of times to train the batch
        """
        states = self.FloatTensor(states)
        actions = self.LongTensor(actions)
        rewards = self.FloatTensor(rewards)
        advantages = self.FloatTensor(advantages)
 
        old_adv_preds, old_v_preds = self.old_model(states)
        old_adv_preds = old_adv_preds.detach()
        old_v_preds = old_v_preds.gather(1, actions.unsqueeze(1)).view(-1,).detach()

        old_logp = F.log_softmax(old_adv_preds, 1).gather(1, actions.unsqueeze(1)).view(-1,).detach()

        for i in range(n_times_batch):
            adv_preds, v_preds = self.model(states)
            probs = F.softmax(adv_preds, 1)
            log_probs = F.log_softmax(adv_preds, 1)

            v_preds = v_preds.gather(1, actions.unsqueeze(1)).view(-1,)

            logp = log_probs.gather(1, actions.unsqueeze(1)).view(-1,).detach()
            entropies = -(probs * log_probs).sum(-1)
            entropy = entropies.mean()

            v_preds_clipped =  old_v_preds + torch.clamp(v_preds - old_v_preds,
                                                         -self.clip_range,
                                                         self.clip_range)

            vf_loss_unclipped = (v_preds - rewards) ** 2
            vf_loss_clipped = (v_preds_clipped - rewards) ** 2
            vf_losses = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped)
            vf_loss = vf_losses.mean()

            p_loss_ratio = torch.exp(logp - old_logp)
            p_loss_unclipped = -advantages * p_loss_ratio
            p_loss_clipped = -advantages * torch.clamp(p_loss_ratio,
                                                       1.0 - self.clip_range,
                                                       1.0 + self.clip_range)
            p_losses = torch.max(p_loss_unclipped, p_loss_clipped)
            p_loss = p_losses.mean()

            losses = p_losses - self.ent_coeff * entropies + self.vf_coeff * vf_losses
            loss = losses.mean()

            if(self.writer is not None):
                self.writer.add_scalar("Train/Policy Loss", p_loss,
                                       self.steps_done)
                self.writer.add_scalar("Train/Value Loss", vf_loss,
                                       self.steps_done)
                self.writer.add_scalar("Train/Entropy", entropy,
                                       self.steps_done)
                self.writer.add_scalar("Train/Loss", loss, self.steps_done)

            self.optimizer.zero_grad()
            loss.backward()

            if(self.max_grad_norm is not None):
                nn.utils.clip_grad_norm(self.parameters(), self.max_grad_norm)

            self.optimizer.step()

        return losses

    def save(self, save_path):
        """
        Creates a save checkpoint of the model at the save path

        save_path : The path to save the checkpoint
        """
        save_dict = {"state_dict" : self.model.state_dict(),
                     "optimizer" : self.optimizer.state_dict(),
                     "steps_done" : self.steps_done}
        torch.save(save_dict, save_path)

    def load(self, load_path):
        """
        Loads the save checkpoint at the given load path

        load_path : The path of the checkpoint to load
        """
        save_state = torch.load(load_path)

        self.model.load_state_dict(save_state["state_dict"])
        self.optimizer.load_state_dict(save_state["optimizer"])
        self.steps_done = save_state["steps_done"]
