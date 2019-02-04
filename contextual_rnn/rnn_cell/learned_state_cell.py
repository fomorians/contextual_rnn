from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from contextual_rnn.rnn_cell import cell_utils


class LearnedStateCell(tf.keras.Model):
    """Forces the state to be learned."""

    def __init__(self, cell, state_fn=None, activation=None):
        """Creates a new LearnedStateCell.

        Args:
            cell: `tf.nn.rnn_cell.RNNCell`-like.
            state_fn: callable that takes a tensor representing the 
                context and returns the nested structure corresponding
                to the nested structure of `cell.state_size`.
            activation: optional activation function for the default
                `state_fn` model.
        """
        super(LearnedStateCell, self).__init__()
        self._cell = cell
        if state_fn is None:
            # TODO(you): Future work.
            state_fn = cell_utils.StateModel(
                self._cell.state_size,
                activation=activation,
                kernel_initializer=tf.initializers.variance_scaling(2.))
        self._state_built = False
        self._state_fn = state_fn

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        return self._cell.state_size

    def zero_state(self, inputs, dtype=None):
        """Creates the 'zero_state' of the cell
        
        Args:
            inputs: Tensor passed to `state_fn`.
            dtype: Unused.

        Returns:
            cell state(s) with nested structure of underlying cell(s).
        """
        del dtype
        assert isinstance(inputs, tf.Tensor)
        inputs.shape.assert_has_rank(2)
        states = self._state_fn(inputs)
        return states

    def call(self, *args, **kwargs):
        return self._cell(*args, **kwargs) 

