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

    @property
    def hyperparameters(self):
        """
        Returns the hyperparameters of the model
        """
        return self._hyperparameters

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

    def find_hyperparams(self):
        """
        Finds hyperparameters that are attributes or optimizer parameters
        """
        attr_dict = self.__dict__

        hyperparam_dict = {}

        # Search for hyperparameters as attributes or optimizer params
        # Set them to the regular float value
        for attr in attr_dict:
            if str(attr_dict[attr].__class__).find("Hyperparameter") != -1:
                if(attr_dict[attr].search):
                    attr_dict[attr].initialize()

                hyperparam_dict[attr] = attr_dict[attr]
                setattr(trainer, attr, attr_dict[attr].item())
            elif str(attr_dict[attr].__class__).find("optim") != -1:
                optim_state_dict = attr_dict[attr].state_dict()
                hyperparam_dict[attr] = optim_state_dict
                param_groups = optim_state_dict["param_groups"][0]

                # Go through the param group and replace all hyperparams
                for param in param_groups:
                    if(str(param_groups[param].__class__).find("Hyperparameter")
                       != -1):
                       if(param_groups[param].search):   
                           param_groups[param].initialize()

                       param_groups[param] = param_groups[param].item()

                attr_dict[attr].load_state_dict(optim_state_dict)

        self._hyperparameters = hyperparam_dict

    def set_hyperparameters(self, hyperparam_dict):
        """
        Sets attributes of the trainer with that appear in the hyperparameters
        dictionary with the value in the dictionary

        hyperparam_dict : The hyperparameter dictionary to set hyperparameters
                          params
        """
        attr_dict = self.__dict__

        self._hyperparameters = {}

        for attr in hyperparam_dict:
            if attr in attr_dict:
                if(str(hyperparam_dict[attr].__class__).find("Hyperparameter")
                   != -1 and hyperparam_dict[attr].search):
                    setattr(self, attr, hyperparam_dict[attr].item())
                    self._hyperparameters[attr] = hyperparam_dict[attr]

                # Otherwise create the state dict for the optimizer
                else:
                    optim_params = hyperparam_dict[attr]
                    param_groups = optim_params["param_group"][0]

                    for param in param_groups:
                        if(str(param_groups[param].__class__).find("Hyperparameter")
                           != -1 and param_groups[param].search):
                            param_groups[param] = param_groups[param].item()

                    attr_dict[attr].load_state_dict(optim_params)
                    self._hyperparameters[attr] = optim_params

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