import tensorflow as tf
import tensorflow.keras as k


def softmax(_logits, axis):
    return tf.exp(_logits) / tf.reduce_sum(tf.exp(_logits), axis, keepdims=True)


@tf.function
def norm(data):
    e = 1e-10
    squared_sum = tf.reduce_sum(tf.square(data), axis=-1)
    return tf.sqrt(squared_sum + e)


@tf.function
def routing_step(_logits, _pre_activation):
    """
    Weight the prediction by routing weights, squash it, and return it
    :param _logits: (batch_size, p_num_caps, num_caps, 1, 1)
    :param _pre_activation: (batch_size, p_num_caps, num_caps, dim_caps, 1)
    :return:
    """
    # softmax of logits over all capsules (such that their sum is 1)
    _prob = softmax(_logits, axis=2)  # shape: (batch_size, p_num_caps, num_caps, 1, 1)
    # calculate _pre_activation based on _prob
    _pre_activation = tf.reduce_sum(_prob * _pre_activation, axis=1,
                                    keepdims=True)  # shape: (batch_size, 1, num_caps, dim_caps, 1)
    return _pre_activation


@tf.function
def routing_loop(_i, _logits, _pre_activation):
    # step 1: find the activation from logits
    _activation = routing_step(_logits, _pre_activation)  # shape: (batch_size, 1, num_caps, dim_caps, 1)
    # step 2: apply squash function over dim_caps
    _activation = squash(_activation, axis=-2)  # shape: (batch_size, 1, num_caps, dim_caps, 1)
    # step 2: find the agreement (dot product) between pre_activation and activation, across dim_caps
    _agreement = tf.reduce_sum(_pre_activation * _activation, axis=-2,
                               keepdims=True)  # shape: (batch_size, p_num_caps, num_caps, 1, 1)
    # step 3: update routing weights based on agreement
    _logits = _logits + _agreement
    # return updated variables
    return _i + 1, _logits, _pre_activation


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


def margin_loss(_y_true, _y_pred, _m_p=0.9, _m_n=0.1, _lambda=0.5):
    """
    Loss Function
    :param _y_true: shape: (None, num_caps)
    :param _y_pred: shape: (None, num_caps)
    :param _m_p: threshold for positive
    :param _m_n: threshold for negative
    :param _lambda: loss weight for negative
    :return: margin loss. shape: (None, )
    """
    p_err = tf.maximum(0., _m_p - _y_pred)  # shape: (None, num_caps)
    n_err = tf.maximum(0., _y_pred - _m_n)  # shape: (None, num_caps)
    p_loss = _y_true * tf.square(p_err)  # shape: (None, num_caps)
    n_loss = (1.0 - _y_true) * tf.square(n_err)  # shape: (None, num_caps)
    loss = tf.reduce_mean(p_loss + _lambda * n_loss, axis=-1)  # shape: (None, )
    return loss


class ConvCaps2D(k.layers.Layer):
    def __init__(self, filters, filter_dims, kernel_size, strides=(1, 1), padding='valid', **kwargs):
        super(ConvCaps2D, self).__init__(**kwargs)
        self.filters = filters
        self.filter_dims = filter_dims
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.conv_layer = ...  # initialize at build()

    def build(self, input_shape):
        self.conv_layer = k.layers.Conv2D(
            filters=self.filters * self.filter_dims,
            kernel_size=self.kernel_size,
            strides=self.strides,
            activation='linear',
            groups=input_shape[1] // self.filter_dims,  # capsule-wise isolated convolution
            padding=self.padding
        )
        self.built = True

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'filter_dims': self.filter_dims,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding
        })
        return config

    def call(self, inputs, *args, **kwargs):
        result = tf.reshape(inputs, (-1, inputs.shape[1], inputs.shape[2], tf.reduce_prod(inputs.shape[3:])))
        result = self.conv_layer(result)
        result = tf.reshape(result, shape=(-1, *result.shape[1:3], self.filters, self.filter_dims))
        return result  # shape: (batch_size, rows, columns, filters, filter_dims)


class DenseCaps(k.layers.Layer):
    def __init__(self, caps, caps_dims, routing_iter, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.caps = caps
        self.caps_dims = caps_dims
        self.routing_iter = routing_iter
        self.input_caps = ...
        self.input_caps_dims = ...
        self.w = ...

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'caps': self.caps,
            'caps_dims': self.caps_dims,
            'routing_iter': self.routing_iter,
            'input_caps': self.input_caps,
            'input_caps_dims': self.input_caps_dims
        })
        return config

    def build(self, input_shape: tf.TensorShape):
        # sanity check
        tf.assert_equal(input_shape.rank, 5, message=f'Expected Tensor of Rank = 5, Got Rank = {input_shape.rank}')
        # define capsule parameters
        self.input_caps = input_shape[1] * input_shape[2] * input_shape[3]
        self.input_caps_dims = input_shape[4]
        # define weights
        self.w = self.add_weight(
            name='w',
            shape=(1, self.input_caps, self.caps, self.caps_dims, self.input_caps_dims),
            dtype=tf.float32,
            initializer=k.initializers.TruncatedNormal()
        )
        self.built = True

    def call(self, inputs, **kwargs):
        # get batch size of input
        batch_size = tf.shape(inputs)[0]
        # reshape input
        inputs = tf.reshape(inputs, (
            batch_size, self.input_caps, 1, 1,
            self.input_caps_dims))  # shape: (batch_size, p_num_caps, 1, 1, p_dim_caps)
        # calculate pre_activation (dot product of w and input over input_caps)
        initial_activation = tf.reduce_sum(self.w * inputs, axis=-1,
                                           keepdims=True)  # shape: (batch_size, p_num_caps, num_caps, dim_caps, 1)
        # dynamic routing
        activation = self.dynamic_routing(initial_activation)  # shape: (batch_size, 1, num_caps, dim_caps, 1)
        # reshape to (None, num_caps, dim_caps) and return
        return tf.squeeze(activation, axis=[1, 4])  # shape: (batch_size, num_caps, dim_caps)

    def dynamic_routing(self, initial_activation):
        """
    Dynamic Routing as proposed in the original paper

    :param initial_activation: shape: (batch_size, p_num_caps, num_caps, dim_caps, 1)
    :return:
    """
        tensor_shape = tf.shape(initial_activation)
        batch_size = tensor_shape[0]
        input_caps = tensor_shape[1]
        caps = tensor_shape[2]
        # define variables
        logits = tf.zeros(shape=(batch_size, input_caps, caps, 1, 1))  # shape: (batch_size, p_num_caps, num_caps, 1, 1)
        # update logits at each routing iteration
        [_, final_logits, _] = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(
            loop_vars=[tf.constant(0), logits, initial_activation],
            cond=lambda i, l, a: i < self.routing_iter,
            body=routing_loop
        ))
        # return activation from the updated logits
        return routing_step(final_logits, initial_activation)  # shape: (batch_size, 1, num_caps, dim_caps, 1)


class FlattenCaps(k.layers.Layer):
    def __init__(self, caps, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(FlattenCaps, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self.caps = caps
        self.input_caps = ...
        self.input_caps_dims = ...
        self.w = ...

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'caps': self.caps,
        })
        return config

    def build(self, input_shape):
        # sanity check
        tf.assert_equal(input_shape.rank, 5, message=f'Expected Tensor of Rank = 5, Got Rank = {input_shape.rank}')
        # define capsule parameters
        self.input_caps = input_shape[1] * input_shape[2] * input_shape[3]
        self.input_caps_dims = input_shape[4]
        # define weights
        self.w = self.add_weight(
            name='w',
            shape=(1, self.caps, self.input_caps, 1),  # (1, c, c_in, 1)
            dtype=tf.float32,
            initializer=k.initializers.TruncatedNormal(),
        )
        self.built = True

    def call(self, inputs, **kwargs):
        inputs = tf.reshape(inputs, (-1, 1, self.input_caps, self.input_caps_dims))  # (b, 1, c_in, d)
        output = tf.reduce_sum(inputs * self.w, axis=-2)  # (b, c, c_in, d) -> (b, c, d)
        return output  # (b, c, d)
