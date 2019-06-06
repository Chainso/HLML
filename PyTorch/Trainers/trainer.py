import torch

from abc import ABC, abstractmethod

class Trainer(ABC):
    def __init__(self, model, device):
        """
        model : The model to train
        device : The device to train on
        """
        self._device = torch.device(device)
        self._model = model.to(self.device)

        self._score = 0

    @property
    def model(self):
        """
        The model the trainer is training
        """
        return self._model

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

    def save(self, save_path):
        """
        Creates a save checkpoint of the model at the save path

        save_path : The path to save the checkpoint
        """
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        """
        Loads the save checkpoint at the given load path

        load_path : The path of the checkpoint to load
        """
        self.model.load_state_dict(torch.load(load_path))