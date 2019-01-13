import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class Model(ABC, nn.Module):
    """
    An abstract base model for any network
    """
    def __init__(self, device):
        """
        Craetes an abstract model
        """
        nn.Module.__init__(self)

        self._device = torch.device(device)
        self.num_batches = 0

    @property
    def device(self):
        """
        The device the model is on
        """
        return self._device

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
