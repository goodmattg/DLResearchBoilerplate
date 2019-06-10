import time


def evaluate(session, model, feed_dict):
    start_time = time.time()
    # feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    test_cost, test_accuracy = session.run(
        [model.loss, model.accuracy], feed_dict=feed_dict
    )
    test_duration = time.time() - start_time
    return test_cost, test_accuracy, test_duration
