from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from contextual_rnn.rnn_cell import fast_weights_cell


@tfe.run_all_tests_in_graph_and_eager_modes
class FastWeightsTest(tf.test.TestCase):

    def test_FastWeightsRNNCell(self):
        inputs = tf.zeros([1, 2])
        initial_states = (tf.zeros([1, 2]), tf.zeros([1, 2, 2]))
        with tf.variable_scope(
                'test_fast_weights_cell', initializer=tf.constant_initializer(0.5)):
            cell = fast_weights_cell.FastWeightsRNNCell(2)
            outputs, states = cell(inputs, initial_states)
        self.evaluate([v.initializer for v in cell.variables])
        self.assertAllClose(
            self.evaluate(outputs), 
            self.evaluate(tf.constant([[0., 0.]])))


if __name__ == '__main__':
    tf.test.main()