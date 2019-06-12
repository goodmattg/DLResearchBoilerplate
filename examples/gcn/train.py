from __future__ import division, print_function

import time
import tensorflow as tf

from utils import *
from evaluate import evaluate
from models import GCN, GCN_Keras

from absl import app
from absl import flags


# def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
#     with tf.GradientTape() as tape:
#         en_outputs = encoder(source_seq, en_initial_states)
#         en_states = en_outputs[1:]
#         de_states = en_states

#         de_outputs = decoder(target_seq_in, de_states)
#         logits = de_outputs[0]
#         loss = loss_func(target_seq_out, logits)

#     variables = encoder.trainable_variables + decoder.trainable_variables
#     gradients = tape.gradient(loss, variables)
#     optimizer.apply_gradients(zip(gradients, variables))

#     return loss

# def training_step(model, inputs):
#     with tf.GradientTape() as tape:


def train(config):

    # Set random seed
    np.random.seed(config.seed)
    tf.compat.v1.set_random_seed(config.seed)

    # Load dataset
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
        config.dataset
    )

    # Some preprocessing
    features = preprocess_features(features)

    train_features = sparse_matrix_to_sparse_tensor(
        preprocess_features(features[train_mask])
    )
    val_features = sparse_matrix_to_sparse_tensor(
        preprocess_features(features[val_mask])
    )
    test_features = sparse_matrix_to_sparse_tensor(
        preprocess_features(features[test_mask])
    )

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

    model = model_constructor(
        input_dim=train_features.shape[1],
        output_dim=y_train.shape[1],
        supports=support,
        num_features_nonzero=train_features.values.shape,
        config=config.model,
    )

    model.fit(
        x=train_features,
        y=tf.boolean_mask(y_train, train_mask),
        epochs=config.training.epochs,
        validation_data=(val_features, tf.boolean_mask(y_val, val_mask)),
        verbose=2,
    )

    # TODO: APPLY MASKING BEFOREHAND before trying to fit everything (i.e. train)

    # Define placeholders
    # placeholders = {
    #     "support": [
    #         tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)
    #     ],
    #     "features": tf.compat.v1.sparse_placeholder(
    #         tf.float32, shape=tf.constant(features[2], dtype=tf.int64)
    #     ),
    #     "labels": tf.compat.v1.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    #     "labels_mask": tf.compat.v1.placeholder(tf.int32),
    #     "dropout": tf.compat.v1.placeholder_with_default(0.0, shape=()),
    #     "num_features_nonzero": tf.compat.v1.placeholder(
    #         tf.int32
    #     ),  # helper variable for sparse dropout
    # }

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
