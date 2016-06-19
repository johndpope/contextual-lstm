import tensorflow as tf


class MatrixFactorizer(object):
    def __init__(self, is_training, config):
        self._config = config

        self._mean_rating = tf.placeholder(tf.float32, name="mean_rating")

        self._input = self.define_input()
        self._output = self.define_output()

        self._cost = self.define_cost()

        if not is_training:
            return

        self._train_op = self.define_training()

    def define_input(self):
        return tf.placeholder(tf.float32, [self.config.num_ratings], name="ratings")

    def define_output(self):
        P = tf.get_variable("P", [self.config.num_users, self.config.rank])
        Q = tf.get_variable("Q", [self.config.rank, self.config.num_items])

        return {'R': tf.matmul(P, Q), 'P': P, 'Q': Q}

    def define_cost(self):
        results = tf.gather(tf.reshape(self.output['R'], [-1]), self.config.user_indices * tf.shape(self.output['R'])[1] + self.config.item_indices)

        #error_op = tf.add(results, self.mean_rating) - self.input
        error_op = results - self.input
        #TODO: Train w/ Cross Entropy?
        sum_squared_error = tf.reduce_sum(tf.square(error_op))
        regularization = self.config.mu * (tf.reduce_sum(tf.square(self.output['P'])) + tf.reduce_sum(tf.square(self.output['Q'])))

        return tf.div(sum_squared_error + regularization, self.config.num_ratings * 2)

    def define_training(self):
        lr = self.config.learning_rate
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        return optimizer.minimize(self.cost, global_step=global_step)

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def mean_rating(self):
        return self._mean_rating

    @property
    def config(self):
        return self._config
