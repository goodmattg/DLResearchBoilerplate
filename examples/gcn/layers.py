import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.initializers import glorot_uniform


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1.0 / keep_prob)


def dot(a, b):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if isinstance(a, tf.SparseTensor):
        res = tf.sparse.sparse_dense_matmul(a, b)
    else:
        res = tf.matmul(a, b)
    return res


class GraphConvolution(Layer):
    def __init__(
        self,
        output_dim,
        supports,
        num_features_nonzero,
        dropout=0.0,
        activation=tf.nn.relu,
        use_bias=False,
        featureless=False,
        **kwargs
    ):
        self.output_dim = output_dim
        self.supports = supports
        self.num_features_nonzero = num_features_nonzero
        self.dropout = dropout
        self.activation = activation
        self.featureless = featureless
        self.use_bias = use_bias

        super(GraphConvolution, self).__init__(**kwargs)

    def build(self, input_shape):

        # Create a trainable weight variable for this layer for each support
        self.weights_per_support = [
            self.add_weight(
                name="weight_support_{}".format(i),
                shape=(input_shape[1], self.output_dim),
                initializer=glorot_uniform(),
                trainable=True,
            )
            for i in range(len(self.supports))
        ]

        # Optionally create trainable bias variable for this layer.
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias", shape=output_dim, initializer="zeros", trainable=True
            )

        super(GraphConvolution, self).build(input_shape)

    def call(self, x):

        if isinstance(x, tf.SparseTensor):
            # Input is sparse, apply sparse dropout
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, self.dropout)

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
