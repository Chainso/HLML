import torch
import torch.nn as nn

class Hyperparameter(nn.Parameter):
    def __new__(cls, data=None, search=True, initializer=None):
        """
        search : Include this hyperparameter in a hyperparameter search
        initializer : The initializer to use for initialization and
                      reinitialization
        """
        return nn.Parameter.__new__(cls, torch.FloatTensor([data]), False)

    def __init__(self, param, name, search=True, initializer=None):
        nn.Parameter.__init__(self)

        self.param_name = name
        self.search = search
        self.initializer = initializer

        self.initialize()

    def __deepcopy__(self, memo):
        result = super(nn.Parameter, self).__deepcopy.__()
        result.search = self.search

    def __repr__(self):
        return "Hyperparameter " + self.param_name + ": " + str(self.item())

    def initialize(self):
        if(self.initializer is not None):
            self.initializer(self)