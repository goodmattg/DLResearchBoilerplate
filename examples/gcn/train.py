from __future__ import division, print_function

import time, datetime
import tensorflow as tf

from utils import *
from models import GCN


def dataset_generator(features, labels, mask):
    while True:
        # Note: workaround to use multiple inputs with Keras Sequential model
        # Mask is concatenated with truth labels, then spliced off in model loss & accuracy
        yield (features, np.column_stack((labels, mask)))


def train(config):

    # Set random seed
    tf.random.set_seed(config.seed)
    np.random.seed(config.seed)

    # Load dataset
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
        config.dataset
    )

    # Feature preprocessing
    norm_feat = preprocess_features(features)
    sparse_norm_feat = sparse_matrix_to_sparse_tensor(norm_feat)

    if config.model.name == "gcn":
        support = [preprocess_adj(adj)]
        num_supports = 1
    elif config.model.name == "gcn_cheby":
        support = chebyshev_polynomials(adj, config.model.max_degree)
        num_supports = 1 + config.model.max_degree
    else:
        raise ValueError("Invalid argument for model: {0}".format(config.model.name))

    model = GCN(
        input_dim=sparse_norm_feat.shape,
        output_dim=y_train.shape[1],
        supports=support,
        num_features_nonzero=tuple(sparse_norm_feat.values.shape.as_list()),
        config=config.model,
    )

    train_generator = dataset_generator(sparse_norm_feat, y_train, train_mask)
    val_generator = dataset_generator(sparse_norm_feat, y_val, val_mask)

    # Train
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=1,
        epochs=config.training.epochs,
        verbose=2,
        validation_data=val_generator,
        validation_steps=1,
    )

    # Evaluate
    test_loss, test_acc = model.test_on_batch(
        sparse_norm_feat, np.column_stack((y_test, test_mask))
    )

    print("Test set results: cost= {:.5f}, accuracy={:.5f}".format(test_loss, test_acc))
