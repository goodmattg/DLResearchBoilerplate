from layers import *
from metrics import *

from tensorflow.keras import optimizers, regularizers, Sequential
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.models import Model


def GCN(input_dim, output_dim, supports, num_features_nonzero, config, **kwargs):
    def loss(self, config):
        def _weight_decay_loss(y_true, y_pred):
            loss = 0
            for weight in self.layers[0].trainable_weights:
                loss += config.weight_decay * tf.nn.l2_loss(weight)
            return loss

        def _cross_entropy_loss(y_true, y_pred):
            labels = y_true[:, :-1]
            mask = y_true[:, -1]
            loss = 0
            loss += masked_softmax_cross_entropy(labels, y_pred, mask)
            return loss

        def _loss(y_true, y_pred):
            wd_loss = _weight_decay_loss(y_true, y_pred)
            ce_loss = _cross_entropy_loss(y_true, y_pred)
            return wd_loss + ce_loss

        return _loss

    def acc(y_true, y_pred):
        labels = y_true[:, :-1]
        mask = y_true[:, -1]
        return masked_accuracy(labels, y_pred, mask)

    model = Sequential()

    model.add(Input(batch_shape=input_dim, sparse=True))

    model.add(
        GraphConvolution(
            output_dim=config.hidden1,
            supports=supports,
            num_features_nonzero=num_features_nonzero,
            activation=tf.nn.relu,
            dropout=config.dropout,
        )
    )

    model.add(
        GraphConvolution(
            output_dim=output_dim,
            supports=supports,
            num_features_nonzero=num_features_nonzero,
            activation=tf.nn.softmax,
            dropout=config.dropout,
        )
    )

    model.compile(
        loss=loss(model, config),
        metrics=[acc],
        optimizer=optimizers.Adam(lr=config.learning_rate),
    )

    return model
