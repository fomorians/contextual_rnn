from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from contextual_rnn.rnn_cell import learned_distribution_state_cell


@tfe.run_all_tests_in_graph_and_eager_modes
class LearnedDistributionStateCellTest(tf.test.TestCase):

    def testZeroState(self):
        cell_fn = tf.nn.rnn_cell.LSTMCell
        inputs = tf.zeros([1, 2])
        with tf.variable_scope(
                'learned_distribution_state_cell', 
                initializer=tf.constant_initializer(0.0)):
            unwrapped_cell = cell_fn(2)
            cell = learned_distribution_state_cell.LearnedDistributionStateCell(
                unwrapped_cell, 
                state_scale_initializer=tf.constant_initializer(0.0))
            initial_states = cell.zero_state(inputs, training=True)
            outputs, states = cell(inputs, initial_states)

        self.evaluate([v.initializer for v in cell.variables])
        self.assertAllClose(
            self.evaluate(outputs), 
            self.evaluate(tf.constant([[0., 0.]])))


if __name__ == '__main__':
    tf.test.main()