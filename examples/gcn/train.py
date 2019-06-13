from __future__ import division, print_function

import time
import tensorflow as tf

from utils import *
from evaluate import evaluate
from models import GCN, GCN_Keras


def train(config):

    # Set random seed
    np.random.seed(config.seed)
    tf.compat.v1.set_random_seed(config.seed)

    # Load dataset
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
        config.dataset
    )

    # Feature preprocessing
    norm_feat = preprocess_features(features)
    sparse_norm_feat = sparse_matrix_to_sparse_tensor(norm_feat)

    if config.model.name == "gcn":
        # Support is sparse representation of normalized adjacency matrix (coords, values)
        # TODO: consider renaming "support"
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_constructor = GCN_Keras
    elif config.model.name == "gcn_cheby":
        support = chebyshev_polynomials(adj, config.model.max_degree)
        num_supports = 1 + config.model.max_degree
        model_constructor = GCN
    elif config.model.name == "dense":
        # TODO: should be able to use Keras OOTB
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_constructor = MLP
    else:
        raise ValueError("Invalid argument for model: {0}".format(config.model.name))

    # import pdb

    # pdb.set_trace()

    model = model_constructor(
        input_dim=sparse_norm_feat.shape,
        output_dim=y_train.shape[1],
        supports=support,
        num_features_nonzero=sparse_norm_feat.values.shape,
        config=config.model,
    )

    print(model.metrics_names)
    for epoch in range(config.training.epochs):
        t = time.time()

        train_loss, train_acc = model.train_on_batch(
            sparse_norm_feat, np.column_stack((y_train, train_mask))
        )
        val_loss, val_acc = model.test_on_batch(
            sparse_norm_feat, np.column_stack((y_val, val_mask))
        )

        print(
            "Epoch: {:04d}, train_loss= {:.5f}, train_acc={:.5f}, val_loss={:.5f}, val_acc={:.5f}, time={:.5f}".format(
                (epoch + 1), train_loss, train_acc, val_loss, val_acc, (time.time() - t)
            )
        )

    # TODO: APPLY MASKING BEFOREHAND before trying to fit everything (i.e. train)

    # Create model
    # model = model_constructor(placeholders, input_dim=features[2][1], logging=True)

    # cost_val = []

    # # Train model
    # for epoch in range(config.epochs):

    #     t = time.time()

    #     # Construct feed dictionary
    #     feed_dict = construct_feed_dict(
    #         features, support, y_train, train_mask, placeholders
    #     )

    #     feed_dict.update({placeholders["dropout"]: config.dropout})

    #     # Training step
    #     outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    #     # Validation
    #     val_feed_dict = construct_feed_dict(
    #         features, support, y_val, val_mask, placeholders
    #     )
    #     cost, acc, duration = evaluate(sess, model, val_feed_dict)
    #     cost_val.append(cost)

    #     # Print results
    #     print(
    #         "Epoch: {:04d}, train_loss= {:.5f}, train_acc={:.5f}, val_loss={:.5f}, val_acc={:.5f}, time={:.5f}".format(
    #             (epoch + 1), outs[1], outs[2], cost, acc, (time.time() - t)
    #         )
    #     )

    #     # Early stopping
    #     if epoch > config.early_stopping and cost_val[-1] > np.mean(
    #         cost_val[-(config.early_stopping + 1) : -1]
    #     ):
    #         print("Early stopping...")
    #         break

    # print("Optimization Finished!")

    # # Evaluation on test set
    # test_feed_dict = construct_feed_dict(
    #     features, support, y_test, test_mask, placeholders
    # )
    # test_cost, test_acc, test_duration = evaluate(sess, model, test_feed_dict)

    # print(
    #     "Test set results: cost= {:.5f}, accuracy={:.5f}, time={:.5f}".format(
    #         test_cost, test_acc, test_duration
    #     )
    # )
