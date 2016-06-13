import unittest

import tensorflow as tf
import numpy as np

from data_sets.io.movielens.mf_context import MfContext
from data_sets.io.movielens.ml_reader import MlReader


class SessionMF(object):
    def __init__(self, config, q):
        self._config = config

        #self._input_data = tf.placeholder(tf.int32, [1, config.num_steps])
        self._features = tf.placeholder(tf.float32, [config.num_steps, config.rank])
        self._q = q

        self._targets = tf.placeholder(tf.int32, [1, config.num_steps])

        self._cost = self.define_cost()

    def define_cost(self):
        outputs = []
        for time_step in range(self.config.num_steps):
            if time_step > 0:
                user_features = tf.reduce_sum(self.features[0:time_step - 1, :], 0, keep_dims=True) / (time_step + 1)
            else:
                user_features = tf.zeros([1, self.config.rank])

            output = tf.matmul(user_features, tf.transpose(self.q))
            outputs.append(output)

        outputs = tf.reshape(tf.concat(1, outputs), [-1, self.config.item_dim])
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
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

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


class SessionMFTest(unittest.TestCase):
    @staticmethod
    def test_session_mf():
        data_path = "data_sets/src/ml-100k"

        mf_context = MfContext(data_path)
        q_dict = mf_context.context_data[0]

        reader = MlReader()

        config = Config()
        config.item_dim = mf_context.config.num_items


        raw_data = reader.raw_data(data_path)
        _, _, data, item_dim = raw_data

        with tf.Graph().as_default(), tf.Session() as session:
            session_mf = SessionMF(config, mf_context.q)
            tf.initialize_all_variables().run()
            costs = 0.0
            num_iter = 0

            # temp = [self.q[item] for item in batch_element]

            for step, (x, y) in enumerate(reader.data_iterator(data, 1, config.num_steps)):
                #features = [q[str(e1)] for e1 in x[0]]
                features = []

                for element in x[0]:
                    for e in q_dict[str(element)]:
                        features.append(e)

                features = np.reshape(features, [-1, 2])
                cost, = session.run([session_mf.cost], {session_mf.targets: y, session_mf.features: features})

                costs += cost
                num_iter += config.num_steps

            return np.exp(costs / num_iter)


if __name__ == "__main__":
    unittest.main()
