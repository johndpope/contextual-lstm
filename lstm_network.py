from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from contextual_lstm_cell import ContextualLSTMCell
from contextual_lstm_cell import ContextualMultiRNNCell


class LSTMNetwork(object):
    def __init__(self, is_training, config):
        self._is_training = is_training
        self._config = config

        self._input_data = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])
        self._context = tf.placeholder(tf.int32, [config.batch_size, config.num_steps, config.context_dim])
        self._targets = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])

        #self._lstm_cell = lstm_cell = self.define_lstm_cell()
        self._lstm_cell = lstm_cell = self.define_contextual_lstm_cell()

        self._initial_state = lstm_cell.zero_state(config.batch_size, tf.float32)

        self._inputs = self.define_input()
        self._outputs, self._final_state = self.define_output(self._initial_state)

        self._cost = self.define_cost()

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        self._train_op = self.define_training()

    def define_input(self):
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [self.config.vocab_size, self.config.hidden_size])
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if self.is_training and self.config.keep_prob < 1.0:
            inputs = tf.nn.dropout(inputs, self.config.keep_prob)

        return inputs

    def define_output(self, state):
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.config.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # TODO: Implement Contextual part
                print(self.context)
                (cell_output, state) = self.lstm_cell(self.inputs[:, time_step, :], self.context, state)
                outputs.append(cell_output)

        return [tf.reshape(tf.concat(1, outputs), [-1, self.config.hidden_size]), state]

    def define_lstm_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size, forget_bias=0.0)
        if self.is_training and self.config.keep_prob < 1.0:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.config.keep_prob)

        return tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.config.num_layers)

    def define_contextual_lstm_cell(self):
        lstm_cell = ContextualLSTMCell(self.config.hidden_size, forget_bias=0.0)
        if self.is_training and self.config.keep_prob < 1.0:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.config.keep_prob)

        return ContextualMultiRNNCell([lstm_cell] * self.config.num_layers)

    def define_cost(self):
        softmax_w = tf.get_variable("softmax_w", [self.config.hidden_size, self.config.vocab_size])
        softmax_b = tf.get_variable("softmax_b", [self.config.vocab_size])

        logits = tf.matmul(self.outputs, softmax_w) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([self.config.batch_size * self.config.num_steps])]
        )

        return tf.reduce_sum(loss) / self.config.batch_size

    def define_training(self):
        trainable_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_vars), self.config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)

        return optimizer.apply_gradients(zip(grads, trainable_vars))

    def assign_lr(self, session, lr):
        session.run(tf.assign(self.lr, lr))

    @property
    def lstm_cell(self):
        return self._lstm_cell

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def lr(self):
        return self._lr

    @property
    def is_training(self):
        return self._is_training

    @property
    def input_data(self):
        return self._input_data

    @property
    def context(self):
        return self._context

    @property
    def targets(self):
        return self._targets

    @property
    def config(self):
        return self._config

