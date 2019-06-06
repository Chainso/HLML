import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from .base_gan import GANModel

class SwapGAN(GANModel):
    """
    Creates the regular GAN model
    """
    def __init__(self, device, gen, disc, swap_size, gen_optim, gen_optim_args,
                 disc_optim, disc_optim_args):
        """
        Creates the regular GAN using the given generator and discriminator

        device : The device for the model to run on
        gen : The generator for the GAN
        disc : The discriminator for the GAN
        swap_size : The size of the block to swap in the real and generated
                    images
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
        self.swap_size = swap_size
        self.summary = None

    def forward(self, inp):
        """
        Will generate an image for the given input noise

        inp : The input noise to generate an image from
        """
        return self.gen(inp)

    def create_summary(self, logs_path):
        """
        Creates a summary that will be written to tensorboard
        """
        if(self.summary is None):
            self.summary = SummaryWriter(logs_path)

    def train_batch(self, data, disc_steps):
        """
        Trains the model for the given batch of data

        data : The data to train the model on
        disc_steps : The number of discriminator steps to train for per
                     per generator
        """
        ones_target = torch.ones(data.shape[0], self.swap_size,
                                 self.swap_size, device = self.device)

        loss_func = nn.BCELoss()

        disc_fake_loss = 0
        disc_real_loss = 0
        disc_loss = 0

        for _ in range(disc_steps):
            self.disc_optim.zero_grad()
            gen_data_detach = self.gen(data).detach()

            swap_out = swap(data, gen_data_detach, self.swap_size, self.device)
            swap_data, swap_gdd, swapped_ones, swapped_zeros = swap_out

            disc_on_gen_detach = self.disc(swap_gdd).squeeze(1)
            disc_on_real = self.disc(swap_data).squeeze(1)

            disc_fake_loss = loss_func(disc_on_gen_detach, swapped_zeros)
            disc_real_loss = loss_func(disc_on_real, swapped_ones)
            disc_loss = disc_fake_loss + disc_real_loss
    
            disc_loss.backward()
            self.disc_optim.step()

        self.gen_optim.zero_grad()

        gen_data = self.gen(data)

        disc_on_gen = self.disc(gen_data).squeeze(1)
        gen_loss = loss_func(disc_on_gen, ones_target)

        gen_loss.backward()
        self.gen_optim.step()

        np_gen_loss = gen_loss.item()
        np_disc_loss = disc_loss.item()

        self.num_batches += 1

        if(self.summary is not None):
            self.summary.add_scalar("Generator Loss", np_gen_loss,
                                    self.num_batches)
            self.summary.add_scalar("Discriminator Real Loss",
                                    disc_real_loss.item(), self.num_batches)
            self.summary.add_scalar("Discriminator Fake Loss",
                                    disc_fake_loss.item(), self.num_batches)
            self.summary.add_scalar("Discriminator Loss", np_disc_loss,
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

def swap(image, other, block_size, device = torch.device("cpu")):
    """
    Swaps two random blocks in the height and width dimension of the given block
    size in the 2 input images, with the block starting on an index such that
    (index + 1) % block_size == 0. The input image must have height and width
    that is divisible by the block size.
    
    Returns swapped image, swapped other, swapped ones target, swapped zeros
    target, ones target

    image : The first image to swap blocks with
    other : The second image to swap blocks with
    block_size : The size of the block to swap
    device : The device to put tensors on
    """
    bs = block_size

    assert (image.shape[-1] == image.shape[-2]) & (image.shape[-1] % bs == 0)
    assert image.shape == other.shape

    rand_bound = image.shape[-1] // bs

    # The block indices
    hb = torch.randint(rand_bound, (1,), dtype = torch.int64).item()
    wb = torch.randint(rand_bound, (1,), dtype = torch.int64).item()

    # Get the target tensors
    img_targ = torch.ones(image.shape[0], bs, bs, device = device)
    other_targ = torch.zeros(image.shape[0], bs, bs, device = device)

    img_targ[:, hb, wb] = 0
    other_targ[:, hb, wb] = 1

    # Get the image indices
    hi = bs * hb
    wi = bs * wb

    # Swap the two blocks
    cloned_img = image.clone().detach().requires_grad_(image.requires_grad)
    cloned_other = other.clone().detach().requires_grad_(other.requires_grad)

    cloned_img[:, :, hi:hi + bs, wi:wi + bs] = other[:, :, hi:hi + bs, wi:wi + bs]
    cloned_other[:, :, hi:hi + bs, wi:wi + bs] = image[:, :, hi:hi + bs, wi:wi + bs]

    return cloned_img, cloned_other, img_targ, other_targ
