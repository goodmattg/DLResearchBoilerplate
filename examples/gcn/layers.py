import tensorflow as tf

from inits import *
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.initializers import glorot_uniform

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=""):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1.0 / keep_prob)


def dot(x, y):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if isinstance(x, tf.SparseTensor):
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(
        self,
        input_dim,
        output_dim,
        placeholders,
        dropout=0.0,
        sparse_inputs=False,
        act=tf.nn.relu,
        bias=False,
        featureless=False,
        **kwargs
    ):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders["dropout"]
        else:
            self.dropout = 0.0

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders["num_features_nonzero"]

        with tf.compat.v1.variable_scope(self.name + "_vars"):
            for i in range(len(self.support)):
                self.vars["weights_" + str(i)] = glorot(
                    [input_dim, output_dim], name="weights_" + str(i)
                )
            if self.bias:
                self.vars["bias"] = zeros([output_dim], name="bias")

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - (1 - self.dropout))

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(
                    x, self.vars["weights_" + str(i)], sparse=self.sparse_inputs
                )
            else:
                pre_sup = self.vars["weights_" + str(i)]

            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars["bias"]

        return self.act(output)


class GraphConvolutionKeras(Layer):
    def __init__(
        self,
        input_dim,
        output_dim,
        supports,
        num_features_nonzero,
        dropout=0.0,
        activation=tf.nn.relu,
        use_bias=False,
        featureless=False,
        **kwargs
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.supports = supports
        self.num_features_nonzero = num_features_nonzero
        self.dropout = dropout
        self.activation = activation
        self.featureless = featureless
        self.use_bias = use_bias

        # helper variable for sparse dropout
        # self.num_features_nonzero = placeholders["num_features_nonzero"]

        super(GraphConvolutionKeras, self).__init__(**kwargs)

    def build(self, input_shape):

        # Create a trainable weight variable for this layer for each support
        self.weights_per_support = [
            self.add_weight(
                name="weight_support_{}".format(i),
                shape=(self.input_dim, self.output_dim),
                initializer=glorot_uniform(seed=None),
                trainable=True,
            )
            for i in range(len(self.supports))
        ]

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias", shape=output_dim, initializer="zeros", trainable=False
            )

        # Create trainable bias variable for this layer.
        super(GraphConvolutionKeras, self).build(
            input_shape
        )  # Be sure to call this at the end

    def call(self, x):

        if isinstance(x, tf.SparseTensor):
            # Input is sparse, apply sparse dropout
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - (1 - self.dropout))

        # Convolve
        supports = list()
        for i in range(len(self.supports)):
            weight_name = "weight_support_{}".format(i)
            if not self.featureless:
                pre_sup = dot(x, self.weights_per_support[i])
            else:
                pre_sup = self.weights_per_support[i]

            support = dot(self.supports[i], pre_sup)
            supports.append(support)
        output = tf.add_n(supports)

        # Bias
        if self.use_bias:
            output += self.bias

        return self.activation(output)
