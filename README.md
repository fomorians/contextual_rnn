# Contextual RNN

This repository contains the code for the paper "[Contextual Recurrent Neural Networks](https://arxiv.org/abs/)"

# Installing

You can install the library with `pip`

```sh
$ git clone https://github.com/fomorians/contextual_rnn.git
$ (cd contextual_rnn; pip install -e .)
```

## Usage

```python
import tensorflow as tf
import contextual_rnn
tf.enable_eager_execution()

cell = tf.nn.rnn_cell.LSTMCell(50)
zero = cell
variable = contextual_rnn.rnn_cell.FreeStateCell(cell)
learned = contextual_rnn.rnn_cell.LearnedStateCell(cell)
learned_distribution = contextual_rnn.rnn_cell.LearnedDistributionStateCell(cell)

batch_size = 10
inputs = tf.zeros([batch_size, 5, 2], dtype=tf.float32)
zero_state = zero.zero_state(
  batch_size, dtype=tf.float32)
variable_state = variable.zero_state(
  batch_size, dtype=tf.float32)
learned_state = learned.zero_state(
  inputs[:, 0], dtype=tf.float32)
learned_distribution_state = learned_distribution.zero_state(
  inputs[:, 0], training=True)
```

## Running Baselines

### ART Task

```sh
python -m contextual_rnn.train_art --job-dir jobs/ --state-type zero --k 8 --seed 42
python -m contextual_rnn.train_art --job-dir jobs/ --state-type learned --k 8 --seed 42
```

### LCD Task

```sh
python -m contextual_rnn.train_lcd --job-dir jobs/ --state-type zero --seed 42
python -m contextual_rnn.train_lcd --job-dir jobs/ --state-type learned-distribution --seed 42
```

## Development

We used [pipenv](https://pipenv.readthedocs.io/en/latest/) to manage dependencies and versions

```sh
pipenv install
pipenv shell
```

## Citation

```
@article{wenkesj2019contextual,
  title={Contextual Recurrent Neural Networks},
  author={Wenke, Sam and Fleming, Jim},
  url={https://arxiv.org/abs/},
  year={2019}
}
```