from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from evaluate import evaluate
from models import GCN, MLP


def train(config):

    # Set random seed
    np.random.seed(config.seed)
    tf.set_random_seed(config.seed)

    Settings pulled from configuration file
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string(
        "dataset", config.dataset, "Dataset string."
    )  # "cora", "citeseer", "pubmed"
    flags.DEFINE_string(
        "model", config.model, "Model string."
    )  # "gcn", "gcn_cheby", "dense"
    flags.DEFINE_float("learning_rate", config.learning_rate, "Initial learning rate.")
    flags.DEFINE_integer("epochs", config.epochs, "Number of epochs to train.")
    flags.DEFINE_integer(
        "hidden1", config.hidden1, "Number of units in hidden layer 1."
    )
    flags.DEFINE_float(
        "dropout", config.dropout, "Dropout rate (1 - keep probability)."
    )
    flags.DEFINE_float(
        "weight_decay", config.weight_decay, "Weight for L2 loss on embedding matrix."
    )
    flags.DEFINE_integer(
        "early_stopping",
        config.early_stopping,
        "Tolerance for early stopping (# of epochs).",
    )
    flags.DEFINE_integer(
        "max_degree", config.max_degree, "Maximum Chebyshev polynomial degree."
    )
    
    # Load and Preprocess Features / Data

    # Define placeholders dictionary
    placeholders = {}

    # Create model

    # Initialize session
    sess = tf.Session()

    # Init variables
    sess.run(tf.global_variables_initializer())

    # Track cost (loss) over time to support
    cost_tracking = []

    # Train model
    for epoch in range(config.epochs):

        t = time.time()

        # Construct training feed dictionary

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Construct validation feed dictionary

        # Validation step

        # Print results
        print(
            "Epoch: {:04d}, train_loss= {:.5f}, train_acc={:.5f}, val_loss={:.5f}, val_acc={:.5f}, time={:.5f}".format(
                (epoch + 1), outs[1], outs[2], cost, acc, (time.time() - t)
            )
        )

        # Early stopping
        early_stop_check = cost_val[-1] > np.mean(
            cost_val[-(config.early_stopping + 1) : -1]
        )
        if epoch > config.early_stopping and early_stop_check:
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # Evaluation on test set

    ## Create test feed dict

    ## Evaluate on test feed dict

    print(
        "Test set results: cost= {:.5f}, accuracy={:.5f}, time={:.5f}".format(
            test_cost, test_acc, test_duration
        )
    )
