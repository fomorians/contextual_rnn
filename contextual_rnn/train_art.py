from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import random
import string
import numpy as np

import tensorflow as tf
from tensorflow.python.util import nest
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp

from contextual_rnn import encoders
from contextual_rnn.rnn_cell import fast_weights_cell
from contextual_rnn.rnn_cell import free_state_cell
from contextual_rnn.rnn_cell import learned_state_cell


class ARTModel(tf.keras.Model):
    """ART task model."""

    def __init__(self, 
                 input_size,
                 wrap_cell_fn, 
                 cell_fn, 
                 zero_state_inputs_fn, 
                 zero_state_fn):
        super(ARTModel, self).__init__()
        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        self._inputs_encode = encoders.Embedding(input_size)
        self._outputs_encode = encoders.OneHotEncoder(input_size)
        self._cells = cell_fn(50)
        self._rnn_cell = wrap_cell_fn(self._cells)
        self._hidden0 = tf.layers.Dense(
            100, 
            activation=tf.nn.relu,
            kernel_initializer=kernel_initializer)
        self._logits0 = tf.layers.Dense(
            self._outputs_encode.shape[-1],
            kernel_initializer=kernel_initializer)
        self._zero_state_inputs_fn = zero_state_inputs_fn
        self._zero_state_fn = zero_state_fn

    def call(self, inputs, training=False):
        embed = self._inputs_encode(inputs)
        state_inputs = self._zero_state_inputs_fn(embed)
        initial_state = self._zero_state_fn(self._rnn_cell, state_inputs)
        hidden, _ = tf.nn.dynamic_rnn(
            self._rnn_cell, 
            embed, 
            initial_state=initial_state,
            dtype=tf.float32)

        hidden = self._hidden0(hidden)
        outputs = self._logits0(hidden)
        return outputs

    def compute_loss(self, inputs, targets, training=False):
        logits = self(inputs, training=training)
        logits = logits[:, -int(targets.shape[1]):]
        dist = tfp.distributions.Categorical(logits=logits)
        nll_loss = tf.losses.compute_weighted_loss(
            -dist.log_prob(targets))
        total_loss = nll_loss
        return total_loss, nll_loss, dist.mode()


class ARTTask(object):
    """
    Implements the associative retrieval task (ART + mART)
    https://arxiv.org/abs/1610.06258
    """

    def __init__(self, chars=list(string.ascii_lowercase)):
        """Create a new ARTTask instance that creates samples from the alphabet `chars`.
        Arguments:
        chars: `list` of `str` that represents the alphabet to sample from. Must not include
            numbers/characters [0-9] or '?'.
        """
        assert not ('?' in chars)
        self._chars = chars
        self._chars_size = len(self._chars) - 1
        self._target_alphabet = list(map(str, list(range(10))))
        self._alphabet = self._chars + self._target_alphabet + ['?']
        self._alphabet_size = len(self._alphabet) - 1
        self._encoder = np.eye(self._alphabet_size + 1)

    @property
    def vocab_size(self):
        return len(self._encoder)

    def create_example(self, k=8, use_modified=False):
        """Creates a single example of length `k`.

        Arguments:
            k: an even `int` that defines the length of the sample space. For example, if `k = 8` and the
                vocab contains `ATCG`, then a sample would look like this: (A9C5G1T3??C, 5).
            use_modified: `bool` that, when `True`, makes samples contiguous alpha-numeric. For example,
                when `k = 8` and the vocab contains `ATCG`, then a sample would look like this:
                (ACTG9513??C, 5). The label is the same as the unmodified version, but the sequence is no
                longer zipped.

        Returns:
            A tuple containing the one-hot encoded values from the vocab. For example, if `k = 8` and the
                vocab contains `ATCG`, then this would return onehot(A9C5G1T3??C, 5), where the
                length of the onehot encoding vectors = len('ACTG') + len([0...9]) + len('?')
                = `vocab_size`.
        """
        q, r = divmod(k, 2)
        assert r == 0 and k > 1 and k < self._alphabet_size, \
            "k must be even, > 1, and < {}".format(self._alphabet_size)

        letters = np.random.choice(range(0, self._chars_size), q, replace=False)
        numbers = np.random.choice(
            range(self._chars_size + 1, self._alphabet_size), q, replace=True)

        if use_modified:
            x = np.concatenate((letters, numbers))
        else:
            x = np.stack((letters, numbers)).T.ravel()

        x = np.append(x, [self._alphabet_size] * 2)
        index = np.random.choice(range(0, q), 1, replace=False)
        x = np.append(x, [letters[index]]).astype('int')
        y = numbers[index]
        x, y = (self._encoder[x], self._encoder[y])
        return np.argmax(x, axis=-1), np.argmax(y, axis=-1)

    def create_datasets(self, num_train_samples, num_eval_samples, k=10):
        train_samples = []
        eval_samples = []

        for _ in range(num_train_samples):
            sample_inputs, sample_targets = self.create_example(k=k)
            train_samples.append((sample_inputs, sample_targets))

        for _ in range(num_eval_samples):
            sample_eval_inputs, sample_eval_targets = self.create_example(k=k)
            eval_samples.append((sample_eval_inputs, sample_eval_targets))

        (sample_inputs, sample_targets) = zip(*train_samples)
        (sample_eval_inputs, sample_eval_targets) = zip(*eval_samples)

        sample_inputs = tf.stack(sample_inputs, axis=0)
        sample_targets = tf.stack(sample_targets, axis=0)
        sample_eval_inputs = tf.stack(sample_eval_inputs, axis=0)
        sample_eval_targets = tf.stack(sample_eval_targets, axis=0)

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (sample_inputs, sample_targets))

        eval_dataset = tf.data.Dataset.from_tensors(
            (sample_eval_inputs, sample_eval_targets))

        return train_dataset, eval_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument(
        '--state-type', 
        choices=[
            'zero',
            'free',
            'learned'], 
        required=True, 
        type=str)
    parser.add_argument('--k', default=8, type=int)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    tf.enable_eager_execution()
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    state_type = args.state_type

    job_dir = os.path.join(args.job_dir, 'art_{}'.format(args.k), state_type)
    global_step = tfe.Variable(0, dtype=tf.int64)

    # Initialize the art task
    task = ARTTask()
    k = args.k

    batch_size = 128
    num_train_samples = 100000
    num_eval_samples = 20000
    num_epochs = 200

    train_dataset, eval_dataset = task.create_datasets(
        num_train_samples, num_eval_samples, k)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.shuffle(num_train_samples // batch_size)
    optimizer = tf.train.AdamOptimizer(1e-3)
    rnn_cell = fast_weights_cell.FastWeightsRNNCell
    job_dir = os.path.join(job_dir, 'fast_weights')
    print(job_dir)

    if state_type == 'zero':
        """Initialize the zero state."""
        task_model = ARTModel( 
            task.vocab_size,
            wrap_cell_fn=(lambda c: c), 
            cell_fn=(lambda n: rnn_cell(n)),
            zero_state_inputs_fn=(lambda inputs: inputs),
            zero_state_fn=(lambda cell, inputs: cell.zero_state(
                inputs.shape[0], dtype=tf.float32)))
    elif state_type == 'free':
        """Initialize the free state."""
        task_model = ARTModel( 
            task.vocab_size,
            wrap_cell_fn=(
                lambda c: free_state_cell.FreeStateCell(c)), 
            cell_fn=(lambda n: rnn_cell(n)),
            zero_state_inputs_fn=(lambda inputs: inputs),
            zero_state_fn=(lambda cell, inputs: cell.zero_state(
                inputs.shape[0], dtype=tf.float32)))
    elif state_type == 'learned':
        """Initialize the learned state."""
        task_model = ARTModel( 
            task.vocab_size,
            wrap_cell_fn=(
                lambda c: learned_state_cell.LearnedStateCell(c)), 
            cell_fn=(lambda n: rnn_cell(n)),
            zero_state_inputs_fn=(lambda inputs: inputs[:, 0]),
            zero_state_fn=(lambda cell, inputs: cell.zero_state(
                inputs)))
    else:
        raise NotImplementedError(state_type)

    # Setup logging and steps.
    summary_writer = tf.contrib.summary.create_file_writer(job_dir)
    summary_writer.set_as_default()

    # Main training/eval loop.
    for epoch in range(num_epochs):
        train_iter = tfe.Iterator(train_dataset) 

        for inputs, targets in train_iter:
            with tf.GradientTape(persistent=True) as tape:
                loss, nll, _ = task_model.compute_loss(
                    inputs, targets, training=True)
            tvars = task_model.trainable_variables
            grads = tape.gradient(loss, tvars)

            # Check grads
            # print(tape.gradient(loss, task_model._rnn_cell._state_fn.trainable_variables))
            # print(tape.gradient(loss, nest.flatten(task_model._rnn_cell._states_structure)))

            optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
            print('epoch {} train loss {}'.format(epoch, nll))
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('loss/nll/train', nll, step=global_step)

        eval_iter = tfe.Iterator(eval_dataset)
        eval_nll = tfe.metrics.Mean()
        eval_accuracy = tfe.metrics.Accuracy()
        for inputs, targets in eval_iter:
            _, nll, predictions = task_model.compute_loss(
                inputs, targets, training=False)
            eval_nll(nll)
            eval_accuracy(targets, predictions)

        eval_nll = eval_nll.result(write_summary=False)
        eval_accuracy = 100. * eval_accuracy.result(write_summary=False)
        print('epoch {} eval accuracy {}'.format(epoch, eval_accuracy))
        print('epoch {} eval loss {}'.format(epoch, eval_nll))
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('loss/nll/eval', eval_nll, step=global_step)
            tf.contrib.summary.scalar('accuracy/eval', eval_accuracy, step=global_step)

        if np.all(eval_accuracy.numpy() == 100.):
            print('exiting task: 100%% accuracy.')
            break