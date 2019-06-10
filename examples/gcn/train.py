from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import GCN, MLP

def train(config):

    # Set random seed
    np.random.seed(config.seed)
    tf.set_random_seed(config.seed)

    # Settings pulled from configuration file
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string("dataset", config.dataset, "Dataset string.")  # "cora", "citeseer", "pubmed"
    flags.DEFINE_string("model", config.model, "Model string.")  # "gcn", "gcn_cheby", "dense"
    flags.DEFINE_float("learning_rate", config.learning_rate, "Initial learning rate.")
    flags.DEFINE_integer("epochs", config.epochs, "Number of epochs to train.")
    flags.DEFINE_integer("hidden1", config.hidden1, "Number of units in hidden layer 1.")
    flags.DEFINE_float("dropout", config.dropout, "Dropout rate (1 - keep probability).")
    flags.DEFINE_float("weight_decay", config.weight_decay, "Weight for L2 loss on embedding matrix.")
    flags.DEFINE_integer("early_stopping", config.early_stopping, "Tolerance for early stopping (# of epochs).")
    flags.DEFINE_integer("max_degree", config.max_degree, "Maximum Chebyshev polynomial degree.")

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(config.dataset)

    # Some preprocessing
    features = preprocess_features(features)

    if config.model == "gcn":
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_constructor = GCN
    elif config.model == "gcn_cheby":
        support = chebyshev_polynomials(adj, config.max_degree)
        num_supports = 1 + config.max_degree
        model_constructor = GCN
    elif config.model == "dense":
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_constructor = MLP
    else:
        raise ValueError("Invalid argument for model: {0}".format(config.model))

    # Define placeholders
    placeholders = {
        "support": [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        "features": tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        "labels": tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        "labels_mask": tf.placeholder(tf.int32),
        "dropout": tf.placeholder_with_default(0., shape=()),
        "num_features_nonzero": tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_constructor(
        placeholders,
        input_dim=features[2][1],
        logging=True)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)


    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(config.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders["dropout"]: config.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)

        # Print results
        print("Epoch:", "%04d" % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
            "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
            "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > config.early_stopping and cost_val[-1] > np.mean(cost_val[-(config.early_stopping+1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
        "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
