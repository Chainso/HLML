import torch.nn as nn

from .utils import forward

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

class ResNetBlock(nn.Module):
    def __init__(self, input_channels, out_channels, batch_norm = True,
                 upsample_scale = None, downsample_scale = None):
        """
        Creates a ResNet block with the given input and output channels, stride
        and padding
    
        input_channels : The number of channels in the input image
        out_channels : The number of output channels in the resulting output
        batch_norm : If the ResNet should use batch normalization
        upsample_scale : If the ResNet should upsample at the end of it
        downsample_scale : If the Resnet should downsample at the end of it
                           (only if up_sampling is None)
        """
        nn.Module.__init__(self)

        self.layers = [None] * 5
        self.layers[0] = nn.Conv2d(input_channels, out_channels, 3, 1, 1)
        self.layers[1] = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.layers[2] = nn.ReLU()
        self.layers[3] = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.layers[4] = nn.BatchNorm2d(out_channels) if batch_norm else None

        self.inp_map = nn.Conv2d(input_channels, out_channels, 1, 1, 0)

        self.up_sampling = None
        self.down_sampling = None

        if(upsample_scale is not None):
            self.up_sampling = nn.ConvTranspose2d(out_channels, out_channels,
                                                 kernel_size = 1,
                                                 stride = upsample_scale,
                                                 output_padding = 1)  
        elif(downsample_scale is not None):
            self.down_sampling = nn.Conv2d(out_channels, out_channels,
                                           kernel_size = 1,
                                           stride = downsample_scale,
                                           padding = 0)

    def forward(self, inp):
        """
        Runs the given input through the network

        inp : The input to run the network on
        """
        resnet = forward(inp, self.layers, [None] * len(self.layers))
        inp_map = self.inp_map(inp)

        res_result = resnet + inp_map
        if(self.up_sampling is not None):
            res_result = self.up_sampling(res_result)
        elif(self.down_sampling is not None):
            res_result = self.down_sampling(res_result)

        return res_result

def resnet(input_channels, out_channels, batch_norm = True,
           upsample_scale = None, downsample_scale = None,
           blocks_per_sample = 1):
    """
    Creates a ResNet network with the given number of input and output channels,
    strides for each ResNet block and paddings for each ResNet block

    input_channels : The number of input channels in the ResNet network
    batch_norm : If the ResNet should use batch normalization
    upsample_scale : If the ResNet should upsample at the end of it
    downsample_scale : If the Resnet should downsample at the end of it
                       (only if up_sampling is None)
    blocks_per_sample : The number of blocks before resizing if using upsampling
                        or downsampling
    """
    tot_channels = [input_channels] + out_channels
    layers = []

    for i in range(len(tot_channels) - 1):
        if(upsample_scale != None or downsample_scale != None):
            for j in range(blocks_per_sample - 1):
                layers.append(ResNetBlock(tot_channels[i], tot_channels[i],
                                          batch_norm))

        layers.append(ResNetBlock(tot_channels[i], tot_channels[i + 1],
                                  batch_norm, upsample_scale,
                                  downsample_scale))

    return nn.ModuleList(layers)
