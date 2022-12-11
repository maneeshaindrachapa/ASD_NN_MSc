import tensorflow as tf
import tensorflow.keras as k


def softmax(_logits, axis):
    return tf.exp(_logits) / tf.reduce_sum(tf.exp(_logits), axis, keepdims=True)


@tf.function
def norm(data):
    e = 1e-10
    squared_sum = tf.reduce_sum(tf.square(data), axis=-1)
    return tf.sqrt(squared_sum + e)


def squash(data, axis=-1):
    """
    Normalize to unit vectors
    :param data: Tensor with rank >= 2
    :param axis: axis over which to squash
    :return:
    """
    e = 1e-10
    squared_sum = tf.reduce_sum(tf.square(data), axis=axis, keepdims=True)
    vec_norm = tf.sqrt(squared_sum + e)
    return squared_sum / (1 + squared_sum) * data / vec_norm


class Mask(k.layers.Layer):

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(Mask, self).__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, inputs, *args, **kwargs):  # input_shape: (None, num_caps, dim_caps)
        # calculate capsule norms
        norms = norm(inputs)  # shape: (None, num_caps)
        # find capsule indices with largest norms
        indices = tf.argmax(norms, axis=-1, output_type=tf.int32)  # shape: (None, )
        # create a mask to apply to input
        mask = tf.expand_dims(tf.one_hot(indices, depth=norms.shape[-1]), axis=-1)  # shape: (None, num_caps, 1)
        # apply mask to input
        return tf.multiply(inputs, mask)  # shape: (None, num_caps, dim_caps)


@tf.function(input_signature=(tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),))
def mask_cid(inputs):
    """
    Select most activated capsule from each instance and return it
    :param inputs: shape: (None, num_caps, dim_caps)
    :return:
    """
    norm_ = norm(inputs)  # shape: (None, num_caps)
    # build index of elements to collect
    i = tf.range(start=0, limit=tf.shape(inputs)[0], delta=1)  # shape: (None, )
    j = tf.argmax(norm_, axis=-1)  # shape: (None, )
    ij = tf.stack([i, tf.cast(j, tf.int32)], axis=1)
    # gather from index and return
    return tf.gather_nd(inputs, ij)
