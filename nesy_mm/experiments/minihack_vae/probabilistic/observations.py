import numpy as np
import tensorflow as tf


class ObservationFunction(tf.Module):
    def __init__(self, lower_bound, upper_bound, grid_size=5):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.grid_size = grid_size + 2

    def __call__(self, samples, training=False, mask=None):
        condition = tf.logical_and(
            samples >= self.lower_bound, samples <= self.upper_bound
        )

        observation = tf.where(condition, 0.0, -np.inf)
        observation = tf.expand_dims(observation, axis=-1)
        return observation
