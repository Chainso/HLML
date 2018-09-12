import torch.nn as nn

from abc import abstractmethod

class MSETrainer(nn.Module):
    """
    A classifier that outputs a distribution of probabilities of the labels
    """
    def __init__(self, input_shape, num_classes, lr, optimizer):
        """
        Creates a multi-class classifier

        input_shape : The shape of the inputs to classify
        num_classes : The number of classes in the labels
        lr : The learning rate of the optimizer
        """
        nn.Module.__init__(self)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr

    @abstractmethod
    def forward(self, inp):
        """
        Returns the output of the network with the given input

        inp : The input to be fed into the network
        """
        pass

    def train_batch(self, inputs, targets):
        """
        Trains the network for a batch of inputs and targets

        inputs : The batch inputs to classify
        targets : The batch target outputs for the network
        """
        preds = self(inputs)

        loss = nn.MSELoss()
        grads = loss(preds, targets)

        self.optimizer.zero_grad()
        grads.backward()
        self.optimizer.step()
