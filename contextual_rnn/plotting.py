from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def plot_states(data, axes, *args, **kwargs):
    t = np.arange(int(data.shape[0]))
    for d, ax in enumerate(axes):
        ax.plot(t, data[..., d].numpy(), *args, **kwargs)


def plot_moments(data, axes, stdevs=2, *args, **kwargs):
    mean, var = tf.nn.moments(data, axes=[0])
    std = tf.sqrt(var)
    t = np.arange(int(data.shape[1]))
    for d, ax in enumerate(axes):
        _ = ax.plot(t, mean[..., d].numpy(), *args, color='r', **kwargs)
        alpha = kwargs.get('alpha', 0.5)
        for i in range(1, stdevs + 1):
            alpha = alpha * 0.8
            lower_bound = (mean[..., d] - i * std[..., d]).numpy()
            upper_bound = (mean[..., d] + i * std[..., d]).numpy()
            ax.fill_between(
                t, lower_bound, upper_bound,
                alpha=alpha, color='r')
