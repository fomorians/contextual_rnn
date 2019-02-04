from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from contextual_rnn.rnn_cell import free_state_cell


@tfe.run_all_tests_in_graph_and_eager_modes
class FreeStateCellTest(tf.test.TestCase):

    def testFreeStateCellZeroState(self):
        cell_fn = tf.nn.rnn_cell.LSTMCell
        inputs = tf.zeros([1, 2])
        with tf.variable_scope(
                'free_state_cell', 
                initializer=tf.constant_initializer(0.5)):
            cell = free_state_cell.FreeStateCell(cell_fn(2))
            initial_states = cell.zero_state(1, tf.float32)
            outputs, states = cell(inputs, initial_states)
        self.evaluate([v.initializer for v in cell.variables])
        self.assertAllClose(
            self.evaluate(outputs), 
            self.evaluate(tf.constant([[0., 0.]])))


if __name__ == '__main__':
    tf.test.main()