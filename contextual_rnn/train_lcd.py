from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import random
import numpy as np

import tensorflow as tf
from tensorflow.python.util import nest
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from contextual_rnn import encoders
from contextual_rnn import plotting
from contextual_rnn.rnn_cell import learned_state_cell
from contextual_rnn.rnn_cell import learned_distribution_state_cell


class ContextualRNN(tf.keras.Model):
    """Contextual RNN."""

    def __init__(self, 
                 states_encode,
                 periods_encode,
                 targets_encode,
                 wrap_cell_fn, 
                 cell_fn, 
                 zero_state_fn,
                 add_periods_to_state=False):
        super(ContextualRNN, self).__init__()
        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)
        self._add_periods_to_state = add_periods_to_state
        self._states_encode = states_encode
        self._periods_encode = periods_encode
        self._targets_encode = targets_encode
        self._cells = cell_fn(128)
        self.rnn_cell = wrap_cell_fn(self._cells)
        self._hidden_state_encode0 = tf.layers.Dense(
            50, 
            activation=tf.nn.relu, 
            kernel_initializer=kernel_initializer)
        self._logits0 = tf.layers.Dense(
            self._states_encode.shape[-1],
            kernel_initializer=kernel_initializer)
        self._zero_state_fn = zero_state_fn
        self._states = None

    def call(self, states, periods, reset_state=True, training=False, testing=False):
        embed = self._states_encode(states)
        if reset_state:
            state_inputs = self._periods_encode(periods)
            if self._add_periods_to_state:
                state_inputs = tf.concat([embed[:, 0], state_inputs[:, 0]], axis=-1)
            else:
                state_inputs = tf.concat([embed[:, 0], state_inputs], axis=-1)
            state_inputs = self._hidden_state_encode0(state_inputs)
            initial_state = self._zero_state_fn(self.rnn_cell, state_inputs, training=training, testing=testing)
            self._states = initial_state
        if self._add_periods_to_state:
            periods = self._periods_encode(periods)
            embed = tf.concat([embed, periods], axis=-1)
        hidden, self._states = tf.nn.dynamic_rnn(
            self.rnn_cell, 
            embed, 
            initial_state=self._states,
            dtype=tf.float32)
        outputs = self._logits0(hidden)
        return outputs

    def compute_loss(self, states, periods, targets, training=False):
        logits = self(states, periods, training=training)[:, -int(targets.shape[1]):]
        mse_loss = tf.losses.mean_squared_error(
            predictions=logits, labels=self._targets_encode(targets))
        total_loss = mse_loss
        return total_loss, mse_loss, logits


def simulate(fn, num_samples, lows, highs, timesteps):
    """Simulate fn with initial states uniformly sampled from `lows` to `highs`."""
    initial_states = tfp.distributions.Uniform(lows, highs).sample(
        [num_samples, 1])
    t = tf.cast(
        tf.tile(tf.reshape(tf.range(timesteps + 1), [1, -1, 1]), [num_samples, 1, 1]), 
        tf.float32)
    states = fn(initial_states, t, timesteps + 1)
    return states


def create_dynamics_fn(period):
    """Returns a function with the given period."""
    def dynamics_fn(state, t, total_t):
        """Simulate linear cosine decay."""
        return tf.train.linear_cosine_decay(
            state,
            t,
            total_t,
            num_periods=period)()
    return dynamics_fn


def create_datasets(train_fns, test_fns, num_samples, num_test_samples, k, lows, highs):
    """Create a train and test dataset."""
    states = []
    targets = []
    periods = []
    for train_fn, period in train_fns:
        sample_states = simulate(train_fn, num_samples, lows, highs, k)
        sample_targets = sample_states[:, 1:]
        sample_states = sample_states[:, :-1]
        sample_targets -= sample_states
        sample_periods = tf.cast(
            tf.tile(tf.reshape(period, [1, 1]), [num_samples, 1]), tf.float32)
        states.append(sample_states)
        periods.append(sample_periods)
        targets.append(sample_targets)
    sample_states = tf.concat(states, axis=0)
    sample_periods = tf.concat(periods, axis=0)
    sample_targets = tf.concat(targets, axis=0)
    dataset = tf.data.Dataset.from_tensor_slices(
        (sample_states, sample_periods, sample_targets))

    min_target = tf.reduce_min(sample_targets, axis=[0, 1])
    max_target = tf.reduce_max(sample_targets, axis=[0, 1])

    states = []
    targets = []
    periods = []
    for test_fn, period in test_fns:
        sample_states = simulate(test_fn, num_test_samples, lows, highs, k)
        sample_targets = sample_states[:, 1:]
        sample_states = sample_states[:, :-1]
        sample_targets -= sample_states
        sample_periods = tf.cast(
            tf.tile(tf.reshape(period, [1, 1]), [num_test_samples, 1]), tf.float32)
        states.append(sample_states)
        periods.append(sample_periods)
        targets.append(sample_targets)
    sample_states = tf.concat(states, axis=0)
    sample_periods = tf.concat(periods, axis=0)
    sample_targets = tf.concat(targets, axis=0)
    return dataset, (sample_states, sample_periods, sample_targets), min_target, max_target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument(
        '--state-type', 
        choices=[
            'learned-distribution',
            'zero'], 
        required=True)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    tf.enable_eager_execution()
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    job_dir = os.path.join(
        args.job_dir, 
        'lcd', 
        args.state_type, 
        'seed_{}'.format(args.seed))
    print(job_dir)

    global_step = tfe.Variable(0, dtype=tf.int64)
    k = 25
    num_epochs = 65
    num_train_samples = 1000
    num_test_samples = 500
    num_generation_samples = 10
    train_fn_periods = [.5, 1.5, 2.5, 3.5, 4.5]
    test_fn_periods = [1., 2., 3., 4.]
    batch_size = 128
    train_dataset, test_dataset, min_target, max_target = create_datasets(
        [(create_dynamics_fn(p), p) for p in train_fn_periods], 
        [(create_dynamics_fn(p), p) for p in test_fn_periods],
        num_train_samples, 
        num_test_samples,
        k,  [2.], [4.])
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.shuffle(num_train_samples // batch_size)
    test_states, test_periods, test_targets = test_dataset

    states_encode = encoders.Scaler([0.], [4.])
    periods_encode = encoders.Scaler([0.5], [4.5])
    targets_encode = encoders.Scaler(min_target, max_target)

    learning_rate = 2e-4
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Create the model.
    add_periods_to_state = False
    if args.state_type == 'learned-distribution':
        forward_model = ContextualRNN( 
            states_encode,
            periods_encode,
            targets_encode,
            wrap_cell_fn=(
                lambda c: learned_distribution_state_cell.LearnedDistributionStateCell(
                    c,
                    state_scale_initializer=learned_distribution_state_cell.softplus_inverse_initializer(.25))), 
            cell_fn=(lambda n: tf.nn.rnn_cell.LSTMCell(n)),
            zero_state_fn=(lambda cell, inputs, training, testing: cell.zero_state(
                inputs, training=training)),
            add_periods_to_state=add_periods_to_state)
    elif args.state_type == 'zero':
        add_periods_to_state = True
        forward_model = ContextualRNN( 
            states_encode,
            periods_encode,
            targets_encode,
            wrap_cell_fn=(lambda c: c),
            cell_fn=(lambda n: tf.nn.rnn_cell.LSTMCell(n)),
            zero_state_fn=(lambda cell, inputs, training, testing: cell.zero_state(
                inputs.shape[0], dtype=tf.float32)),
            add_periods_to_state=add_periods_to_state)
    else:
        raise NotImplementedError()

    checkpoint = tf.train.Checkpoint(
        forward_model=forward_model,
        optimizer=optimizer,
        global_step=global_step)
    checkpoint_path = tf.train.latest_checkpoint(job_dir)
    if checkpoint_path:
        print('Restoring checkpoint from {}'.format(checkpoint_path))
        checkpoint.restore(checkpoint_path)

    # Setup logging and steps.
    summary_writer = tf.contrib.summary.create_file_writer(job_dir)
    summary_writer.set_as_default()

    if add_periods_to_state:
        test_periods = tf.tile(
            tf.expand_dims(test_periods, axis=1), 
            [1, test_states.shape[1], 1])

    # Main training/test loop.
    for epoch in range(num_epochs):
        # Train the model.
        train_iter = tfe.Iterator(train_dataset) 
        for states, periods, targets in train_iter:
            if add_periods_to_state:
                periods = tf.tile(tf.expand_dims(periods, axis=1), [1, states.shape[1], 1])
            with tf.GradientTape(persistent=True) as tape:
                loss, nll, _ = forward_model.compute_loss(
                    states, periods, targets, training=True)
            tvars = forward_model.trainable_variables
            grads = tape.gradient(loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, 10.)

            optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
            print('epoch {} train loss {}'.format(epoch, nll))
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('loss/nll/train', nll, step=global_step)
                tf.contrib.summary.scalar(
                    'learning_rate', 
                    learning_rate() if callable(learning_rate) else learning_rate, 
                    step=global_step)

        checkpoint.save(os.path.join(job_dir, 'ckpt'))

        loss, nll, _ = forward_model.compute_loss(
            test_states, test_periods, test_targets)
        print('epoch {} eval loss {}'.format(epoch, nll))
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('loss/nll/eval', nll, step=global_step)

        for i in range(len(test_fn_periods)):
            sample_idx = i * num_test_samples
            test_outputs = []
            state = test_states[sample_idx, 0]

            if add_periods_to_state:
                period = tf.reshape(test_periods[sample_idx, 0], [1, 1, 1])
                period = tf.tile(period, [num_generation_samples, 1, 1])
            else:
                period = tf.reshape(test_periods[sample_idx], [1, 1])
                period = tf.tile(period, [num_generation_samples, 1])

            # Tiles states and inputs.
            state = tf.reshape(state, [1, 1, -1])
            state = tf.tile(state, [num_generation_samples, 1, 1])
            test_outputs.append(state)
            state += targets_encode.inverse(
                forward_model(state, period, testing=True))
            test_outputs.append(state)
            for _ in range(1, k):
                state += targets_encode.inverse(
                    forward_model(state, period, testing=True, reset_state=False))
                test_outputs.append(state)
            test_outputs = tf.concat(test_outputs, axis=1)

            f, axis = plt.subplots(1)
            plotting.plot_states(test_states[sample_idx], [axis], 'k--')
            for particle_states in tf.unstack(test_outputs, axis=0):
                plotting.plot_states(particle_states, [axis])
            plotting.plot_moments(test_outputs, [axis], 1)
            with tf.gfile.Open(
                os.path.join(
                    job_dir, 
                    'moments_{}_{}'.format(i, epoch + 1) + '.png'), 'w+') as tf_file:
                f.savefig(tf_file)
            plt.close(f)
