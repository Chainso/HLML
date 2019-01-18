import torch
import torch.nn as nn

def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, z_dim, image_channels, num_init_filters, num_hidden):
        nn.Module.__init__(self)

        self.z_dim = z_dim
        self.image_channels = image_channels

        hidden_filt = num_init_filters
        conv_t_blocks = []

        for _ in range(num_hidden):
            out_filts = hidden_filt // 2
            conv_t_blocks.append(self.conv_t_block(hidden_filt, out_filts, 4, 2,
                                                   1))
            hidden_filt = out_filts
    
        self.layers = nn.Sequential(
            self.conv_t_block(z_dim, num_init_filters, 4, 1, 0),
            *conv_t_blocks,
            nn.ConvTranspose2d(hidden_filt, image_channels, 4, 2, 1,
                               bias=False),
            nn.Tanh()
            )

    def forward(self, inp):
        if(len(inp.shape) > 2):
            inp = torch.rand(inp.shape[0], self.z_dim).to(inp.device)

        res_inp = inp.view(-1, self.z_dim, 1, 1)

        return self.layers(res_inp)

    def conv_t_block(self, filt_in, filt_out, kernel_size, stride, padding):
        return nn.Sequential(nn.ConvTranspose2d(filt_in, filt_out, kernel_size,
                                                stride, padding, bias=False),
                             nn.BatchNorm2d(filt_out),
                             nn.ReLU(inplace = True)
                             )

class Discriminator(nn.Module):
    def __init__(self, image_channels, num_init_filters, num_hidden):
        nn.Module.__init__(self)

        self.image_channels = image_channels

        hidden_filt = num_init_filters
        conv_blocks = []

        for _ in range(num_hidden):
            out_filts = hidden_filt * 2
            conv_blocks.append(self.conv_block(hidden_filt, out_filts, 4, 2, 1))
            hidden_filt = out_filts

        self.layers = nn.Sequential(
            nn.Conv2d(image_channels, num_init_filters, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace = True),
            *conv_blocks,
            nn.Conv2d(hidden_filt, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            )

    def forward(self, inp):
        return self.layers(inp)

    def conv_block(self, filt_in, filt_out, kernel_size, stride, padding):
        return nn.Sequential(nn.Conv2d(filt_in, filt_out, kernel_size, stride,
                                       padding, bias=False),
                             nn.BatchNorm2d(filt_out),
                             nn.LeakyReLU(0.2, inplace = True)
                             )
if(__name__ == "__main__"):
    from torch.optim import Adam

    from PyTorch.data.image_data_processor import ImageDataProcessor
    from PyTorch.SL.GAN.noisy_gan import NoisyGAN
    from PyTorch.SL.GAN.gan_runner import GANRunner

    # Generator Parameters
    z_dim = 100
    image_channels = 3
    gen_num_init_filt = 512
    gen_num_hidden = 3
    gen_params = (z_dim, image_channels, gen_num_init_filt, gen_num_hidden)
    gen = Generator(*gen_params)

    # Discriminator Parameters
    disc_num_init_filt = 64
    disc_num_hidden = 3
    disc_params = (image_channels, disc_num_init_filt, disc_num_hidden)
    disc = Discriminator(*disc_params)
    
    # GAN parameters
    device = "cuda"
    optim = Adam
    optim_args = {"lr" : 2e-3, "betas" : (0.5, 0.999)}
    gan_params = (device, gen, disc, optim, optim_args, optim, optim_args)
    gan = NoisyGAN(*gan_params).to(torch.device(device))
    gan.apply(weights_init)
    gan.train()

    # Create the tensorboard summary
    logs_path = "./logs"
    gan.create_summary(logs_path)

    # Data Processor Parameters
    data_path = "./data/celebA"
    resized_image = 64
    data_proc_params = (data_path, resized_image, device)
    data_proc = ImageDataProcessor(*data_proc_params)

    # GANRunner Parameters
    static_noise = torch.randn(10, z_dim, device = torch.device(device))
    gan_runner_params = (gan, data_proc, static_noise, device)
    gan_runner = GANRunner(*gan_runner_params)

    # Training Parameters
    model_save_path = "./DCGAN Models/dcgan_celebA.torch"
    model_test_path = "./DCGAN Tests/celebA Tests"
    epochs = 100
    batch_size = 128
    save_interval = 1
    training_params = (model_save_path, model_test_path, epochs, batch_size,
                       save_interval)

    # Testing parameters
    testing_params = (model_test_path,)

    gan_runner.train(*training_params)
