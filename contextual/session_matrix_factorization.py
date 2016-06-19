from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from contextual.matrix_factorization import MatrixFactorization
from data_sets.io.movielens.ml_reader import MlReader


class SessionMF(object):
    def __init__(self, config, q):
        self._config = config

        #self._input_data = tf.placeholder(tf.int32, [1, config.num_steps])
        self._features = tf.placeholder(tf.float32, [config.num_steps, config.rank])
        self._q = q

        self._mean_value = tf.placeholder(tf.float32)

        self._targets = tf.placeholder(tf.int32, [1, config.num_steps])

        self._cost = self.define_cost()

    def define_cost(self):
        outputs = []
        for time_step in range(self.config.num_steps):
            if time_step > 0:
                user_features = tf.reduce_sum(self.features[0:time_step, :], 0, keep_dims=True) / time_step
            else:
                # Initialize with random small numbers to prevent log(0)
                user_features = tf.random_uniform([1, self.config.rank], minval=0.01, maxval=0.01)

            #output = tf.matmul(user_features, tf.transpose(self.q)) + self.mean_value
            output = tf.matmul(user_features, tf.transpose(self.q))
            sum_over_output = tf.reduce_sum(output)
            output = tf.div(output, sum_over_output)
            #output = tf.log(output)

            outputs.append(output)

        self._outputs = outputs = tf.reshape(tf.concat(1, outputs), [-1, self.config.item_dim])

        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [outputs],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([self.config.num_steps])]
        )

        return tf.reduce_sum(loss)

    @property
    def config(self):

        return self._config

    @property
    def features(self):
        return self._features

    @property
    def targets(self):
        return self._targets

    @property
    def mean_rating(self):
        return self._mean_rating

    @property
    def q(self):
        return self._q

    @property
    def cost(self):
        return self._cost


class Config():
    num_steps = 10
    item_dim = 10000
    rank = 2


class SessionMFTest():
    @staticmethod
    def test_session_mf():
        data_path = "../data_sets/src/ml-100k"

        mf = MatrixFactorization(data_path)
        Q = mf.Q

        reader = MlReader()

        config = Config()
        config.item_dim = mf.config.num_items
        config.rank = mf.config.rank

        raw_data = reader.raw_item_data(data_path)
        _, _, data, item_dim = raw_data

        with tf.Graph().as_default(), tf.Session() as session:
            session_mf = SessionMF(config, Q)
            tf.initialize_all_variables().run()
            costs = 0.0
            num_iter = 0

            # temp = [self.q[item] for item in batch_element]

            for step, (x, y) in enumerate(reader.item_iterator(data, 1, config.num_steps)):
                #features = [q[str(e1)] for e1 in x[0]]
                features = []

                for element in x[0]:
                    for e in Q[element - 1]:
                        features.append(e)

                features = np.reshape(features, [-1, config.rank])
                cost, outputs = session.run([session_mf.cost, session_mf._outputs], {session_mf.targets: y, session_mf.features: features, session_mf._mean_value: mf._mean_value})

                #for element in outputs:
                 #   if np.isnan(element).any() or np.count_nonzero(element) < len(element):
                  #      print(element)

                costs += cost
                num_iter += config.num_steps

            perplexity = np.exp(costs / num_iter)
            print("Test Perplexity: %.3f" % perplexity)
