def forward(inp, layers, activations):
    """
    Computes the output of the given network and activations

    layers : A list of layers for the network
    activations : A list of activation functions
    """
    assert len(layers) == len(activations)

    output = inp

    for layer, activation in zip(layers, activations):
        if(layer is not None):
            output = layer(output)

        if(activation is not None):
            output = activation(output)

    return output

def to_device(tensor, device):
    """
    Convert the device of a tensor

    tensor : The tensor to be converted
    device : The device to store the tensor on

    Returns the tensor on the given device
    """
    if(device == "cpu"):
        tensor = tensor.cpu()
    elif(device == "cuda"):
        tensor = tensor.cuda()

    return tensor

def normalize(x, epsilon):
    """
    Normalizes a given tensor

    x : The given tensor
    epsilon : The epsilon to add to the standard deviation
    """
    return ((x - x.mean()) / (x.std() + epsilon))
