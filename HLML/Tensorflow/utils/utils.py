import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.contrib.layers import xavier_initializer
from scipy import signal

def gaussian_noise(inpu, mean=0.0, stddev=0.1):
    noise = tf.random_normal(tf.shape(inpu), mean=mean,
                             stddev=stddev, dtype=tf.float32)

    return inpu + noise

def trainable_lstm_state(batch_size, state_size, name):
    lstm_state = []

    i = 1
    for state_tuple in state_size:
        var_c = tf.get_variable(name + "_cell" + str(i) + "-c", shape=state_tuple.c,
                                initializer=xavier_initializer())

        var_h = tf.get_variable(name + "_cell" + str(i) + "-h", shape=state_tuple.h,
                                initializer=xavier_initializer())

        state_c = tf.reshape(tf.tile(var_c, [batch_size]), (batch_size, state_tuple.c)) 
        state_h = tf.reshape(tf.tile(var_h, [batch_size]), (batch_size, state_tuple.h))

        lstm_state.append(rnn.LSTMStateTuple(state_c, state_h))
        i += 1

    return tuple(lstm_state)

def cudnn_trainable_lstm_state(num_layers, batch_size, num_units, name):
    state_c = tf.get_variable(name + "_cell-c", shape=(num_layers, num_units),
                              initializer=xavier_initializer())
    state_c = tf.reshape(tf.tile(state_c, [batch_size, 1]), (num_layers, batch_size, num_units))

    state_h = tf.get_variable(name + "_cell-h", shape=(num_layers, num_units),
                              initializer=xavier_initializer())
    state_h = tf.reshape(tf.tile(state_h, [batch_size, 1]), (num_layers, batch_size, num_units))

    return tuple([state_c, state_h])

def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

def factorized_noise(inputs, outputs):
    noise1 = signed_sqrt(tf.random_normal((inputs, 1)))
    noise2 = signed_sqrt(tf.random_normal((1, outputs)))

    return tf.matmul(noise1, noise2)


def signed_sqrt(values):
    return tf.sqrt(tf.abs(values)) * tf.sign(values)

def take(vectors, indices):
    """
    For a batch of vectors, take a single vector component
    out of each vector.
    Args:
      vectors: a [batch x dims] Tensor.
      indices: an int32 Tensor with `batch` entries.
    Returns:
      A Tensor with `batch` entries, one for each vector.
    """
    return tf.gather_nd(vectors, tf.stack([tf.range(tf.shape(vectors)[0]), indices], axis=1))

def normalize(x, epsilon):
    """
    Normalizes a given tensor

    x : The given tensor
    epsilon : The epsilon to add to the standard deviation
    """
    return ((x - x.mean()) / (x.std() + epsilon))
