from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import unittest
import numpy as np
import tensorflow as tf

import ptb_reader
import ml_reader

from lstm_network import LSTMNetwork
flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS


def main():
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to data directory")

    if FLAGS.reader == "ptb":
        reader = ptb_reader
    elif FLAGS.reader == "lastfm":
        reader = lastfm_reader
    elif FLAGS.reader == "ml":
        reader = ml_reader
    else:
        reader = ptb_reader

    raw_data = reader.raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, num_elements = raw_data

    context = reader.context_data(FLAGS.data_path)

    config = get_config()
    config.vocab_size = num_elements
    config.context_dim = len(context[str(1)])
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    eval_config.vocab_size = num_elements

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model_training = LSTMNetwork(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            model_valid = LSTMNetwork(is_training=False, config=config)
            model_test = LSTMNetwork(is_training=False, config=eval_config)

        tf.initialize_all_variables().run()

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            model_training.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(model_training.lr)))
            train_perplexity = run_epoch(session, model_training, train_data, model_training.train_op, context, reader,
                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, model_valid, valid_data, tf.no_op(), reader)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, model_test, test_data, tf.no_op(), reader)
        print("Test Perplexity: %.3f" % test_perplexity)


def run_epoch(session, model, data, eval_op, context, reader, verbose=False):
    epoch_size = ((len(data) // model.config.batch_size) - 1) // model.config.num_steps
    start_time = time.time()
    costs = 0.0
    num_iter = 0
    state = model.initial_state.eval()

    print(context)

    for step, (x, y) in enumerate(reader.data_iterator(data, model.config.batch_size, model.config.num_steps)):
        c = []
        for e0 in x:
            c.append([context[str(e1)] for e1 in e0])

        cost, state, _ = session.run([model.cost, model.final_state, eval_op],
                                     {model.input_data: x,
                                      model.targets: y,
                                      model.context: c,
                                      model.initial_state: state})

        costs += cost
        num_iter += model.config.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / num_iter),
                   num_iter * model.config.batch_size / (time.time() - start_time)))

    return np.exp(costs / num_iter)


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 10
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    context_dim = 18


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class LSTMTest(unittest.TestCase):
    def test_lstm(self):
        FLAGS.data_path = "data/ml-100k"
        FLAGS.model = "small"
        FLAGS.reader = "ml"
        main()

if __name__ == "__main__":
    unittest.main()
    #tf.app.run()