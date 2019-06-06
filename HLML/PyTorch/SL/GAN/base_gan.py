from HLML.PyTorch.base_model import Model
from abc import ABC, abstractmethod

class GANModel(Model, ABC):
    """
    An abstract GAN model
    """
    def __init__(self, device, gen, disc):
        """
        Creates an abstract GAN model

        device : The device for the model to run on
        gen : The generator for the GAN
        disc : The discriminator for the GAN
        """
        Model.__init__(self, device)

        self.gen = gen
        self.disc = disc