from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import rpy2.robjects as robjects

from sklearn.cross_validation import train_test_split
from rpy2.robjects import numpy2ri

from matrix_factorizer import MatrixFactorizer
from context import Context


class MfContext(Context):
    def __init__(self, data_path=None):
        robjects.r['load'](os.path.join(data_path, "ratings-100k.RData"))
        ratings = numpy2ri.ri2py(robjects.r['ratings_df'])
        ratings_tr, ratings_val = train_test_split(ratings)
        rating_values_tr = np.array([rating[2] for rating in ratings_tr], dtype=np.float32)
        rating_values_eval = np.array([rating[2] for rating in ratings_val], dtype=np.float32)
        mean_rating = np.mean(rating_values_tr)
        max_steps = 10
        self._config = config_tr = Config(ratings)
        config_eval = Config(ratings)

        with tf.Graph().as_default(), tf.Session() as session:
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = MatrixFactorizer(True, config_tr)
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                model_eval = MatrixFactorizer(False, config_eval)

            tf.initialize_all_variables().run()

            # Run the graph and see how we're doing on every 500th iteration.
            for i in range(max_steps):
                if i % 500 == 0:
                    res = session.run([model.cost],
                                      {model.mean_rating: mean_rating,
                                       model.input: ratings}
                                      )
                    av_err = res[0]
                    print("Training - Cost: %s" % av_err)

                    res = session.run([model_eval.cost],
                                      {model_eval.mean_rating: mean_rating,
                                       model_eval.input: ratings}
                                      )
                    av_err_eval = res[0]
                    print("Evaluation - Cost: %s" % av_err_eval)
                else:
                    session.run(model.train_op,
                                {model.mean_rating: mean_rating,
                                 model.input: ratings}
                                )

            self._q = Qt = tf.transpose(model.output['Q']).eval()

            #TODO: Correct?
            self._context_data = {str(i): Qt[i] for i in range(len(Qt))}

    @property
    def context_data(self):
        return self._context_data, len(self._context_data[str(0)])

    @property
    def q(self):
        return self._q

    @property
    def config(self):
        return self._config


class Config(object):
    def __init__(self, ratings):
        self.learning_rate = 0.01
        self.mu = 0.1
        self.rank = 2
        self.num_ratings = len(ratings)
        self.user_indices = [np.int32(rating[0]) for rating in ratings]
        self.item_indices = [np.int32(rating[1]) for rating in ratings]

        self.num_users = len(f7(self.user_indices))
        self.num_items = len(f7(self.item_indices))


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]