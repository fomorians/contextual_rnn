from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util import nest


class Projection(tf.keras.Model):
    """Projects to arbitrary shape."""

    def __init__(self, size, activation=None, kernel_initializer=None):
        super(Projection, self).__init__()
        self._size = None
        if isinstance(size, tf.TensorShape):
            self._size = size
            new_size = 1
            for s in size:
                new_size *= s
            self._layer = tf.layers.Dense(
                new_size, 
                activation=activation,
                kernel_initializer=kernel_initializer)
        else:
            self._layer = tf.layers.Dense(
                size, 
                activation=activation,
                kernel_initializer=kernel_initializer)

    def call(self, inputs):
        outputs = self._layer(inputs)
        if isinstance(self._size, tf.TensorShape):
            return tf.reshape(
                outputs, 
                inputs.shape.as_list()[:-1] + self._size.as_list())
        return outputs


class StateModel(tf.keras.Model):
    """Project the state(s) shape(s) through dense layers."""

    def __init__(self, state_size, activation=None, kernel_initializer=None):
        super(StateModel, self).__init__()
        self._layers = tf.contrib.checkpoint.List([])

        def create_loc(size):
            layer = Projection(
                size, 
                activation=activation,
                kernel_initializer=kernel_initializer)
            self._layers.append(layer)
            return layer

        self._cell_project0 = tf.contrib.framework.nest.map_structure(
            create_loc, state_size)

    def call(self, inputs):
        def project(cell_project):
            return cell_project(inputs)
        return tf.contrib.framework.nest.map_structure(
            project, self._cell_project0)
