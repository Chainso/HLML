import torch.nn as nn

def conv_network(input_channels, out_channels, kernel_sizes, strides,
                 paddings):
    """
    Creates a convolution network with the given parameters

    input_channels : The number of channels in the input image
    out_channels : The number of output channels in each convolution layer
    kernel_sizes : The sizes of the kernels in each convolution layer
    strides : The strides of the convolution in each layer
    paddings : The paddings of the convolution in layer
    """
    tot_channels = [input_channels] + out_channels
    layers = []

    for i in range(len(tot_channels) - 1):
        layers.append(nn.Conv2d(tot_channels[i], tot_channels[i + 1],
                                kernel_sizes[i], strides[i], paddings[i]))

    return nn.ModuleList(layers)

def inverse_conv_network(input_channels, out_channels, kernel_sizes, strides,
                         paddings):
    """
    Creates an inverse convolution network with the given parameters

    input_channels : The number of channels in the input image
    out_channels : The number of output channels in each convolution layer
    kernel_sizes : The sizes of the kernels in each convolution layer
    strides : The strides of the convolution in each layer
    paddings : The paddings of the convolution in layer
    """
    tot_channels = [input_channels] + out_channels
    layers = []

    for i in range(len(tot_channels) - 1):
        layers.append(nn.ConvTranspose2d(tot_channels[i], tot_channels[i + 1],
                                         kernel_sizes[i], strides[i],
                                         paddings[i]))

    return nn.ModuleList(layers)

def resnet_block(input_channels):
    pass
    