from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util import nest
import tensorflow_probability as tfp

from contextual_rnn.rnn_cell import cell_utils


def softplus_inverse_initializer(scale):
    def fn(shape, dtype, *args, **kwargs): 
        return tfp.distributions.softplus_inverse(
            scale * tf.ones(shape, dtype=dtype))
    return fn


class LearnedDistributionStateCell(tf.keras.Model):
    """Forces the state to be a parameterized distribution."""

    def __init__(self, 
                 cell, 
                 state_fn=None, 
                 activation=None,
                 state_scale_initializer=softplus_inverse_initializer(.01)):
        super(LearnedDistributionStateCell, self).__init__()
        self._cell = cell
        if state_fn is None:
            # TODO(you): Future work.
            state_fn = cell_utils.StateModel(
                self._cell.state_size,
                activation=activation,
                kernel_initializer=tf.initializers.variance_scaling(2.))
        self._state_built = False
        self._state_variables = tf.contrib.checkpoint.List([])
        self._state_scale_initializer = state_scale_initializer
        self._state_structure = None
        self._state_fn = state_fn

    def _build_state(self):
        def set_state_distribution(zero_state_shape):
            var_name = 'state_{}'.format(len(self._state_variables))
            variable = tf.get_variable(
                name=var_name,
                shape=zero_state_shape,
                initializer=self._state_scale_initializer)
            self._state_variables.append(variable)
            return variable

        self._state_structure = nest.map_structure(
            set_state_distribution, self._cell.state_size)
        self._state_built = True

    def build(self, *args, **kwargs):
        if not self._state_built:
            self._build_state()

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        return self._cell.state_size

    def _call_states(self, inputs, training=False):
        if not self._state_built:
            self._build_state()

        def sample_states(state_scale, state_loc):
            batch_size = state_loc.shape[0]
            shaped_state_loc = tf.reshape(state_loc, [batch_size, -1])
            shaped_state_scale = tf.reshape(state_scale, [-1])

            state_posteriors = tfp.distributions.MultivariateNormalDiag(
                shaped_state_loc, tf.nn.softplus(shaped_state_scale))

            state_sample = tf.cond(
                tf.identity(training), 
                lambda: state_posteriors.sample(), 
                lambda: state_posteriors.mode())
            state_sample = tf.reshape(state_sample, state_loc.shape)
            return state_sample

        state_locs = self._state_fn(inputs)
        states = nest.map_structure(
            sample_states, 
            self._state_structure,
            state_locs)
        return states

    def zero_state(self, inputs, dtype=None, training=False):
        """Creates the 'zero_state' of the cell, with a twist.
        
        Args:
            inputs: Tensor passed to `state_fn`.
            dtype: Unused.
            training: `sample` from the posterior when `training` is set, 
                else `mode`.

        Returns:
            cell state(s) with nested structure of underlying cell(s).
        """
        del dtype
        assert isinstance(inputs, tf.Tensor)
        inputs.shape.assert_has_rank(2)
        states = self._call_states(inputs, training=training)
        return states

    def call(self, *args, **kwargs):
        return self._cell(*args, **kwargs) 
