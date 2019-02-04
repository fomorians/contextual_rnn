from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.util import nest


class FreeStateCell(tf.keras.Model):
    """Forces the state(s) to be trainable variables."""

    def __init__(self, cell):
        """Creates a new FreeStateCell.

        Args:
            cell: `tf.nn.rnn_cell.RNNCell`-like.
        """
        super(FreeStateCell, self).__init__()
        self._cell = cell
        self._state_built = False
        self._state_variables = tf.contrib.checkpoint.List([])
        self._states_structure = None

    def _build_state(self, dtype):        
        def set_state_variable(zero_state_shape):
            var_name = 'state_{}'.format(len(self._state_variables))
            variable = tf.get_variable(
                shape=zero_state_shape,
                initializer=tf.zeros_initializer(dtype=tf.float32),
                name=var_name)
            self._state_variables.append(variable)
            return variable

        self._states_structure = nest.map_structure(
            set_state_variable, self._cell.state_size)
        self._state_built = True

    def build(self, *args, **kwargs):
        if not self._state_built:
            self._build_state(tf.float32)

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        return self._cell.state_size

    def zero_state(self, batch_size, dtype):
        if not self._state_built:
            self._build_state(dtype)

        def tile_state(state):
            return tf.tile(
                state[None, ...], 
                [batch_size] + [1] * state.shape.ndims)

        return nest.map_structure(
            tile_state, self._states_structure)

    def call(self, *args, **kwargs):
        return self._cell(*args, **kwargs) 
