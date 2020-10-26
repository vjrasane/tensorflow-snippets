from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_probability as tfp
import tensorflow as tf

initial_distribution = tfp.distributions.Categorical(probs=[0.8, 0.2])
transition_distribution = tfp.distributions.Categorical(probs=[
  [0.7, 0.3],
  [0.2, 0.8]
])
observation_distribution = tfp.distributions.Normal(loc=[0., 15.], scale=[5., 10.])

model = tfp.distributions.HiddenMarkovModel(
  initial_distribution=initial_distribution,
  transition_distribution=transition_distribution,
  observation_distribution=observation_distribution,
  num_steps=7
)

mean = model.mean()

with tf.compat.v1.Session() as sess:
  print(mean.numpy())