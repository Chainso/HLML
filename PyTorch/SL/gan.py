import torch
import torch.nn as nn

from copy import deepcopy

class CycleGAN(nn.Module):
    """
    A general Cycle-Consistent Generative Adversial Network
    """
    def __init__(self, generator, discriminator, cycle_coeff, lr, optimizer):
        """
        Creates the CycleGAN

        generator : The generator network
        discriminator : The discriminator network
        cycle_coeff : The coefficient of the cycle loss
        lr : The learning rate for the optimizer
        optimizer : The optimizer for the network
        """
        nn.Module.__init__(self)

        self.for_gen = generator
        self.for_disc = discriminator

        self.back_gen = deepcopy(generator)
        self.back_disc = deepcopy(discriminator)

        self.cycle_coeff = cycle_coeff

        self.optimizer = optimizer(self.parameters(), lr=lr)

    def train_batch(self, inp, target):
        """
        Trains the CycleGAN for one batch of inputs and targets

        inp : The input batch
        target : The target batch
        """
        ones_tensor = torch.ones(len(inp))
        zeros_tensor = torch.zeros(len(inp))

        targ_pred = self.for_gen(inp)

        for_real_disc = self.for_disc(inp)
        for_fake_disc = self.for_disc(targ_pred.detach())

        for_disc_real_loss = nn.BCELoss()(for_real_disc, ones_tensor)
        for_disc_fake_loss = nn.BCELoss()(for_fake_disc, zeros_tensor)
        for_disc_loss = for_disc_real_loss + for_disc_fake_loss

        for_gen_loss = nn.BCELoss()(for_fake_disc.detach(), ones_tensor)

        inp_pred = self.back_gen(targ_pred)

        back_real_disc = self.back_disc(target)
        back_fake_disc = self.back_disc(inp_pred.detach())

        back_disc_real_loss = nn.BCELoss()(back_real_disc, ones_tensor)
        back_disc_fake_loss = nn.BCELoss()(back_fake_disc, zeros_tensor)
        back_disc_loss = back_disc_real_loss + back_disc_fake_loss

        back_gen_loss = nn.BCELoss()(back_fake_disc.detach(), ones_tensor)

        cycle_disc = self.back_disc(inp_pred)

        cycle_loss = nn.BCELoss()(cycle_disc, ones_tensor)

        total_loss = (self.cycle_coeff * cycle_loss + for_disc_loss +
                      for_gen_loss + back_disc_loss + back_gen_loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
