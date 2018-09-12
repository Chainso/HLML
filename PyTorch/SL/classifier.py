import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod

class MultiClassClassifier(nn.Module):
    """
    A classifier that outputs a distribution of probabilities of the labels
    """
    def __init__(self, input_shape, num_classes, lr):
        """
        Creates a multi-class classifier

        input_shape : The shape of the inputs to classify
        num_classes : The number of classes in the labels
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

    def probabilities(self, inp):
        """
        Returns the class probabilities of the input

        inp : The input to be fed into the network
        """
        return F.softmax(self.forward(inp)).cpu().data.numpy()

    def train_batch(self, inputs, targets):
        """
        Trains the network for a batch of inputs and targets

        inputs : The batch inputs to classify
        targets : The batch target outputs for the network
        """
        preds = self(inputs)

        loss = nn.CrossEntropyLoss()
        grads = loss(preds, torch.max(targets, 1)[1])

        self.optimizer.zero_grad()
        grads.backward()
        self.optimizer.step()
