
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe


class Embedding(tf.keras.Model):
    
    def __init__(self, num_inputs, num_outputs=100):
        super(Embedding, self).__init__()
        self.num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._embed0 = tfe.Variable(
            tf.zeros([num_inputs, self._num_outputs], tf.float32))

    @property
    def shape(self):
        return [self._num_outputs]

    def call(self, inputs):
        return tf.nn.embedding_lookup(self._embed0, inputs)


class OneHotEncoder(tf.keras.Model):
    
    def __init__(self, num_outputs):
        super(OneHotEncoder, self).__init__()
        self._num_outputs = num_outputs

    @property
    def shape(self):
        return [self._num_outputs]

    def call(self, inputs):
        return tf.cast(tf.one_hot(inputs, self.num_outputs, axis=-1), tf.int32)

    def inverse(self, inputs):
        return tf.cast(tf.argmax(inputs, axis=-1), tf.int32)


class Scaler(tf.keras.Model):

    def __init__(self, min_value, max_value):
        super(Scaler, self).__init__()
        self.min_value = tf.cast(min_value, tf.float32)
        self.max_value = tf.cast(max_value, tf.float32)

    @property
    def shape(self):
        return self.min_value.shape

    def call(self, inputs, weights=1., **kwargs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        inputs = (inputs - self.min_value) / (self.max_value - self.min_value)
        inputs = (inputs * 2.) - 1.
        return inputs * weights

    def inverse(self, inputs, weights=1., **kwargs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        inputs = (inputs + 1.) / 2.
        inputs = (inputs * (self.max_value - self.min_value) + self.min_value)
        inputs = tf.clip_by_value(inputs, self.min_value, self.max_value)
        return inputs * weights
