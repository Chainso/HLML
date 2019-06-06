import torch

from abc import ABC, abstractmethod

class Trainer(ABC):
    def __init__(self, model, device):
        """
        model : The model to train
        device : The device to train on
        """
        self._device = torch.device(device)
        self.model = model.to(self.device)

        self._score = 0

    @property
    def device(self):
        """
        The device the model is on
        """
        return self._device

    @property
    def score(self):
        """
        The score of the last evaluation of the model
        """
        return self._score

    @abstractmethod
    def train_batch(self, *args):
        """
        Trains the model for a single batch
        """
        pass

    @abstractmethod
    def train(self, epochs, save_path=None, save_interval=1, logs_path=None,
              *args):
        """
        Trains the model for the number of epochs given

        epochs : The number of epochs to train for
        save_path : The path to save the model to, None if you do not wish to
                    save
        save_interval : If a save path is given, the number of epochs in between
                        each save
        logs_path : The path to save logs to
        args : Any additional arguments that may be required
        """
        pass

    @abstractmethod
    def eval(self, *args):
        """
        Evaluates the model on the data
        """
        pass