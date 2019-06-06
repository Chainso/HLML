import tensorflow as tf

def conv_net(inpu, filters, filter_sizes, strides, paddings, activations,
             name):
    # All the all_layers in the resnet
    all_layers = []
    all_layers.append(inpu)

    i = 0

    # Build the number of convolution layers given
    for filt, filt_size, stride, padding, activation in zip(filters,
                                                            filter_sizes,
                                                            strides, paddings,
                                                            activations):
        all_layers.append(tf.layers.conv2d(all_layers[-1], filt, filt_size,
                                           stride, padding,
                                           activation = activation,
                                           name = name + str(i)))

        i += 1

    return all_layers[-1]

def resnet_block(inpu, num_filters, name="resnet"):
    conv1 = tf.layers.conv2d(inpu, num_filters, (3, 3), (1, 1),
                             padding="SAME", activation=tf.nn.leaky_relu)

    conv2 = tf.layers.conv2d(conv1, num_filters, (3, 3), (1, 1),
                             padding="SAME", activation=None)

    return tf.nn.leaky_relu(conv2 + inpu)

def resnet(inpu, num_filters, num_blocks, name):
    # All the all_layers in the resnet
    all_layers = []
    all_layers.append(inpu)

    # Build the number of resnet blocks given
    for i in range(num_blocks):
        all_layers.append(resnet_block(all_layers[-1], num_filters,
                                       name + str(i)))

    return all_layers[-1]
