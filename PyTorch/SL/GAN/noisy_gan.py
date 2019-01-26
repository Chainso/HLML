import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from .base_gan import GANModel

class NoisyGAN(GANModel):
    """
    Creates the regular GAN model
    """
    def __init__(self, device, gen, disc, gen_optim, gen_optim_args, disc_optim,
                 disc_optim_args):
        """
        Creates the regular GAN using the given generator and discriminator

        device : The device for the model to run on
        gen : The generator for the GAN
        disc : The discriminator for the GAN
        gen_optim : The optimizer of the generator
        gen_optim_args : The arguments (including learning rate) of the
                         generator's optimizer not including the parameters
        disc_optim : The optimizer of the discriminator
        disc_optim_args : The arguments (including learning rate) of the
                          discriminator's optimizer not including the parameters 
        """
        GANModel.__init__(self, device, gen, disc)

        self.gen_optim = gen_optim(self.gen.parameters(), **gen_optim_args)
        self.disc_optim = disc_optim(self.disc.parameters(), **disc_optim_args)
        self.writer = None

    def create_summary(self, logs_path):
        """
        Creates a summary that will be written to tensorboard
        """
        if(self.writer is None):
            self.writer = SummaryWriter(logs_path)

    def train_batch(self, data, disc_steps):
        """
        Trains the model for the given batch of data

        data : The data to train the model on
        disc_steps : The number of discriminator steps to train for per
                     per generator
        """
        ones_target = torch.ones(data.shape[0], 1, device = self.device)
        zeros_target = torch.zeros(data.shape[0], 1, device = self.device)

        loss_func = nn.BCELoss()

        disc_fake_loss = 0
        disc_real_loss = 0
        disc_loss = 0

        for _ in range(disc_steps):
            self.disc_optim.zero_grad()
            gen_data_detach = self.gen(data).detach()
    
            disc_on_gen_detach = self.disc(gen_data_detach)
            disc_on_real = self.disc(data)
    
            disc_fake_loss = loss_func(disc_on_gen_detach, zeros_target)
            disc_real_loss = loss_func(disc_on_real, ones_target)
            disc_loss = disc_fake_loss + disc_real_loss
    
            disc_loss.backward()
            self.disc_optim.step()

        self.gen_optim.zero_grad()

        gen_data = self.gen(data)

        disc_on_gen = self.disc(gen_data)
        gen_loss = loss_func(disc_on_gen, ones_target)

        gen_loss.backward()
        self.gen_optim.step()

        np_gen_loss = gen_loss.item()
        np_disc_loss = disc_loss.item()

        self.num_batches += 1

        if(self.writer is not None):
            self.writer.add_scalar("Generator Loss", np_gen_loss,
                                   self.num_batches)
            self.writer.add_scalar("Discriminator Real Loss",
                                   disc_real_loss.item(), self.num_batches)
            self.writer.add_scalar("Discriminator Fake Loss",
                                   disc_fake_loss.item(), self.num_batches)
            self.writer.add_scalar("Discriminator Loss", np_disc_loss,
                                   self.num_batches)

        return np_gen_loss, np_disc_loss

    def save(self, save_path):
        """
        Creates a save checkpoint of the model at the save path

        save_path : The path to save the checkpoint
        """
        torch.save(self.state_dict(), save_path)

    def load(self, load_path):
        """
        Loads the save checkpoint at the given load path

        load_path : The path of the checkpoint to load
        """
        self.load_state_dict(torch.load(load_path))

class ControlledNoise(nn.Module):
    """
    A module to replace a random block of data with noise
    """
    def __init__(self, lower_bound, upper_bound, block_size, device):
        """
        Creates the controlled noisy module to create noise for an input image.
        Will replace a block of the denoted size with noise, with the block
        starting on an index such that (index + 1) % block_size == 0.

        lower_bound : The lower bound for the noise
        upper_bound : The upper bound for the noise
        block_size : The size of the noisy block
        device : The device for the noise
        """
        nn.Module.__init__(self)

        self.lb = lower_bound
        self.ub = upper_bound
        self.block_size = block_size
        self.device = torch.device(device)

    def forward(self, img):
        """
        Will replace a block of the denoted size with noise, with the block
        starting on an index such that (index + 1) % block_size == 0. The input
        image must have height and width that is divisible by the block size.

        image : The image to inject noise into
        """
        assert (img.shape[-1] == img.shape[-2]) & (img.shape[-1] / self.block_size
                                                   == img.shape[-1] // self.block_size)

        # img.shape[1] is the number of input channels
        noise = ((self.ub + self.lb) * torch.rand(img.shape[1], self.block_size,
                                                  self.block_size,
                                                  device = self.device) + self.lb)

        rand_bound = img.shape[-1] // self.block_size
        noise_indices = self.block_size * torch.randint(rand_bound,
                                                        (img.shape[0],),
                                                        device = self.device)

        img_indices = torch.stack(noise_indices, noise_indices, dim = 1)
        noisy_image = img[noise_indices