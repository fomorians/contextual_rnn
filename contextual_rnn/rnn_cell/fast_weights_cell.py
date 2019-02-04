from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import tensor_shape, dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib import layers


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

_FastWeightsStateTuple = collections.namedtuple("FastWeightsStateTuple", ("c", "a"))


class FastWeightsStateTuple(_FastWeightsStateTuple):
    """Tuple used by FastWeights Cells for `state_size`, `zero_state`, and output state.
    Stores two elements: `(c, A)`, in that order. Where `c` is the hidden state
    and `A` is the fast weights state.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (c, a) = self
        if c.dtype != a.dtype:
            raise TypeError("Inconsistent internal state: {} vs {}".format(
                            (str(c.dtype), str(a.dtype))))
        return c.dtype


class FastWeightsRNNCell(rnn_cell_impl.LayerRNNCell):

    """
    Original Fast Weights
    https://arxiv.org/abs/1610.06258
    """

    def __init__(self,
                num_units,
                use_layer_norm=True,
                activation=nn_ops.relu,
                fast_learning_rate=.5,
                fast_decay_rate=0.95,
                use_bias=True,
                reuse=None,
                name=None):
        super(FastWeightsRNNCell, self).__init__(_reuse=reuse, name=name)
        self._num_units = num_units
        self._activation = activation
        self._use_layer_norm = use_layer_norm
        self._fast_learning_rate = fast_learning_rate
        self._fast_decay_rate = fast_decay_rate
        self._use_bias = use_bias

    @property
    def state_size(self):
        return FastWeightsStateTuple(
            self._num_units, tensor_shape.TensorShape(
                [self._num_units, self._num_units]))

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        self._kernel_w = self.add_variable(
            "{}_w".format(_WEIGHTS_VARIABLE_NAME),
            [self.output_size, self.output_size],
            dtype=self.dtype,
            initializer=init_ops.identity_initializer(gain=.05))
        self._kernel_c = self.add_variable(
            "{}_c".format(_WEIGHTS_VARIABLE_NAME),
            [inputs_shape[1], self.output_size],
            dtype=self.dtype,
            initializer=init_ops.identity_initializer(gain=.05))
        if self._use_bias:
            self._bias_c = self.add_variable(
                "{}_c".format(_BIAS_VARIABLE_NAME),
                [self.output_size],
                dtype=self.dtype)

    def call(self, inputs, state, training=False):
        hidden_state, fast_weights = state

        batch_size = array_ops.shape(fast_weights)[0]
        add = math_ops.add
        scalar_mul = math_ops.scalar_mul

        slow = array_ops.expand_dims(
            add(
                math_ops.matmul(hidden_state, self._kernel_w),
                nn_ops.bias_add(
                    math_ops.matmul(inputs, self._kernel_c), self._bias_c)),
                1)
        hidden_state = self._activation(slow)

        fast_weights = add(
            scalar_mul(self._fast_decay_rate, fast_weights),
            scalar_mul(self._fast_learning_rate, math_ops.matmul(
                array_ops.transpose(hidden_state, [0, 2, 1]), hidden_state)))

        h = array_ops.identity(hidden_state)
        inner = add(slow, math_ops.matmul(h, fast_weights))
        h = self._activation(
            layers.layer_norm(inner)
            if self._use_layer_norm else inner)
        hidden_state = gen_array_ops.reshape(h, [batch_size, self._num_units])
        return hidden_state, FastWeightsStateTuple(hidden_state, fast_weights)