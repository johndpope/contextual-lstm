from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import unittest
import numpy as np
import tensorflow as tf

from ptb_reader import PtbReader
from ml_reader import MlReader
from lastfm_reader import LastfmReader
from lstm_network import LSTMNetwork


def test_network(data_set):
    reader, data_path = get_setup(data_set)

    raw_data = reader.raw_data(data_path)
    train_data, valid_data, test_data, item_dim = raw_data

    context, context_dim = reader.context_data(data_path)

    config = get_config()
    config.item_dim = item_dim
    config.context_dim = context_dim
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    eval_config.item_dim = item_dim

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model_training = LSTMNetwork(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            model_valid = LSTMNetwork(is_training=False, config=config)
            model_test = LSTMNetwork(is_training=False, config=eval_config)

        tf.initialize_all_variables().run()

        for i in range(config.max_max_epoch):
            training_epoch(model_training, session, train_data, context, config, reader, i)
            validation_epoch(model_valid, session, valid_data, context, reader, i)
        validation_epoch(model_test, session, test_data, context, reader)


def training_epoch(model, session, data, context, config, reader, epoch):
    lr_decay = config.lr_decay ** max(epoch - config.max_epoch, 0.0)
    model.assign_lr(session, config.learning_rate * lr_decay)

    print("Epoch: %d Learning rate: %.3f" % (epoch + 1, session.run(model.lr)))
    perplexity = run_epoch(session, model, data, model.train_op, context, reader, verbose=True)
    print("Epoch: %d Train Perplexity: %.3f" % (epoch + 1, perplexity))


def validation_epoch(model, session, data, context, reader, epoch=None):
    perplexity = run_epoch(session, model, data, tf.no_op(), context, reader)
    if epoch is not None:
        print("Epoch: %d Valid Perplexity: %.3f" % (epoch + 1, perplexity))
    else:
        print("Test Perplexity: %.3f" % perplexity)


def run_epoch(session, model, data, eval_op, context, reader, verbose=False):
    epoch_size = ((len(data) // model.config.batch_size) - 1) // model.config.num_steps
    start_time = time.time()
    costs = 0.0
    num_iter = 0
    state = model.initial_state.eval()

    for step, (x, y) in enumerate(reader.data_iterator(data, model.config.batch_size, model.config.num_steps)):
        c = [[context[str(e1)] for e1 in e0] for e0 in x]
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
    return Config()


def get_setup(data_set):
    if data_set == "ptb":
        return PtbReader(), "data/ptb"
    elif data_set == "lastfm":
        return LastfmReader(), "data/lastfm"
    elif data_set == "ml":
        return MlReader(), "data/ml-100k"
    else:
        return PtbReader(), "data/ptb"


class Config(object):
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
    item_dim = 10000
    context_dim = 18


def test_reader(reader, data_path):
    reader.raw_data(data_path)


class LSTMTest(unittest.TestCase):
    @staticmethod
    def test_lstm():
        test_network("ml")
        # test_network("lastfm")
        # test_network("ptb")

        # test_reader(MlReader(), "data/ml-100k")
        # test_reader(LastfmReader(), "data/lastfm")
        # test_reader(PtbReader(), "data/ptb")


if __name__ == "__main__":
    unittest.main()
