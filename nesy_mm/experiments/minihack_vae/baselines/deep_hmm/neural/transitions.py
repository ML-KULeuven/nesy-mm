import tensorflow as tf
import einops as E

from keras.layers import *


class NeuralTransition(tf.keras.Model):
    def __init__(self, grid_size, log_space):
        super().__init__()
        self.grid_size = grid_size + 2
        self.log_space = log_space

        self.transition = tf.keras.Sequential()
        self.transition.add(Dense(64, activation="relu"))
        self.transition.add(Dense(32, activation="relu"))
        self.transition.add(Dense(self.grid_size * 2))
        self.transition.add(Reshape((2, self.grid_size)))

    def __call__(self, inputs, training=False, mask=None):
        samples = inputs[0]  # (b, d, n)
        action = inputs[1]  # (b)

        n = samples.shape[-1]

        samples = tf.one_hot(samples, depth=self.grid_size)
        samples = E.rearrange(samples, "b d n o -> (b n) (d o)")

        action = tf.one_hot(action, depth=4)
        actions = E.repeat(action, "b d -> (b n) d", n=n)

        x = tf.concat([samples, actions], axis=-1)
        x = self.transition(x)
        x = E.rearrange(x, "(b n) d o -> b d n o", n=n)

        if self.log_space:
            x = tf.nn.log_softmax(x, axis=-1)
        else:
            x = tf.nn.softmax(x, axis=-1)
            x = tf.clip_by_value(x, 1e-6, 1 - 1e-6)
            x = tf.math.log(x)
        return x

    def call(self, inputs, training=False, mask=None):
        return self.__call__(inputs, training=training, mask=mask)
