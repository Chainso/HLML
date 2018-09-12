import tensorflow as tf
import numpy as np

from .utils import factorized_noise

def dense_network(inpu, num_outs, activations, name, dropout=False,
                  keep_prob = 0.5):
    # All the layers in the dense network
    all_layers = []
    all_layers.append(inpu)

    # Build the number of dense layers given
    for i in range(len(num_outs)):
        layer = tf.layers.dense(all_layers[-1], num_outs[i],
                                activations[i],
                                name = name + str(i + 1))

        if(dropout and i < len(num_outs) - 1):
            layer = tf.nn.dropout(layer, keep_prob)

        all_layers.append(layer)

    return all_layers[-1]

def noisy_network(inpu, num_outs, activations, sigmas, name):
    # All the layers in the dense network
    all_layers = []
    all_layers.append(inpu)

    # Build the number of dense layers given
    for i in range(len(num_outs)):
        layer = noisy_layer(all_layers[-1], num_outs[i], activations[i],
                            sigmas[i], name = name + str(i + 1))

        all_layers.append(layer)

    return all_layers[-1]

def noisy_layer(inputs, units, activation=None, sigma0=0.5,
                kernel_initializer=None, name=None, reuse=None):
    """
    Apply a factorized Noisy Net layer.
    See https://arxiv.org/abs/1706.10295.
    Args:
      inputs: the batch of input vectors.
      units: the number of output units.
      activation: the activation function.
      sigma0: initial stddev for the weight noise.
      kernel_initializer: initializer for kernels. Default
        is to use Gaussian noise that preserves stddev.
      name: the name for the layer.
      reuse: reuse the variable scope.
    """
    num_inputs = inputs.get_shape()[-1].value
    stddev = 1 / np.sqrt(num_inputs)
    activation = activation if activation is not None else (lambda x: x)
    if kernel_initializer is None:
        kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)

    with tf.variable_scope(None, default_name=(name or 'noisy_layer'), reuse=reuse):
        weight_mean = tf.get_variable('weight_mu',
                                      shape=(num_inputs, units),
                                      initializer=kernel_initializer)
        bias_mean = tf.get_variable('bias_mu',
                                    shape=(units,),
                                    initializer=tf.zeros_initializer())
        stddev *= sigma0
        weight_stddev = tf.get_variable('weight_sigma',
                                        shape=(num_inputs, units),
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        bias_stddev = tf.get_variable('bias_sigma',
                                      shape=(units,),
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
        bias_noise = tf.random_normal((units,), dtype=bias_stddev.dtype.base_dtype)
        weight_noise = factorized_noise(num_inputs, units)

        return activation(tf.matmul(inputs, weight_mean + weight_stddev * weight_noise) +
                          bias_mean + bias_stddev * bias_noise)