import torch.nn as nn

class Hyperparameter(nn.Parameter):
    def __new__(cls, data=None, search=True, initializer=None):
        """
        search : Include this hyperparameter in a hyperparameter search
        initializer : The initializer to use for initialization and
                      reinitialization
        """
        nn.Parameter.__new__(cls, data, False)

        self.search = search
        self.initializer = initializer

    def __init__(self):
        nn.Parameter.__init__(self)

        self.initialize()

    def __deepcopy__(self, memo):
        result = super(nn.Parameter, self).__deepcopy.__()
        result.search = self.search

    def __repr__(self):
        return "Hyper" + super(nn.Parameter, self).__repr__()

    def initialize(self):
        if(self.initializer is not None):
            self.initializer(self)