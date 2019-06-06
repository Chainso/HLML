import torch
import torch.nn as nn

def linear_network(inp_units, units):
    """
    Creates a linear network with the given number of units per layer and
    activation per layer

    units : The number of output units in each layer
    
    """
    tot_units = [inp_units] + units
    layers = []

    for i in range(len(tot_units) - 1):
        layers.append(nn.Linear(tot_units[i], tot_units[i + 1]))

    return nn.ModuleList(layers)
