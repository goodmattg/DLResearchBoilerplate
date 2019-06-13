from layers import *
from metrics import *

from tensorflow.keras import optimizers, regularizers, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders["features"]
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders["labels"].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate
        )

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(
            self.outputs, self.placeholders["labels"], self.placeholders["labels_mask"]
        )

    def _accuracy(self):
        self.accuracy = masked_accuracy(
            self.outputs, self.placeholders["labels"], self.placeholders["labels_mask"]
        )

    def _build(self):

        self.layers.append(
            GraphConvolution(
                input_dim=self.input_dim,
                output_dim=FLAGS.hidden1,
                placeholders=self.placeholders,
                act=tf.nn.relu,
                dropout=True,
                sparse_inputs=True,
                logging=self.logging,
            )
        )

        self.layers.append(
            GraphConvolution(
                input_dim=FLAGS.hidden1,
                output_dim=self.output_dim,
                placeholders=self.placeholders,
                act=lambda x: x,
                dropout=True,
                logging=self.logging,
            )
        )

    def predict(self):
        return tf.nn.softmax(self.outputs)


def GCN_Keras(input_dim, output_dim, supports, num_features_nonzero, config, **kwargs):

    model = Sequential()

    model.add(Input(batch_shape=input_dim, sparse=True))

    model.add(
        GraphConvolutionKeras(
            output_dim=config.hidden1,
            supports=supports,
            num_features_nonzero=num_features_nonzero,
            activation=tf.nn.relu,
            dropout=config.dropout,
        )
    )

    # TODO: MISSING REGULARIZATION OF FIRST LAYER
    # model.add(regularizers.l2(config.weight_decay))

    # TODO: Take Dropout out of GCN layer

    model.add(
        GraphConvolutionKeras(
            output_dim=output_dim,
            supports=supports,
            num_features_nonzero=num_features_nonzero,
            activation=tf.identity,
            dropout=config.dropout,
        )
    )

    model.add(Dense(output_dim, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        optimizer=optimizers.Adam(lr=config.learning_rate),
    )

    return model
