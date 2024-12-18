import einops as E
import tensorflow as tf

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
        agent_loc = inputs[0]
        agent_loc = tf.cast(agent_loc, tf.keras.backend.floatx())

        n = agent_loc.shape[-1]

        action = inputs[1]
        action = E.repeat(action, "b d -> b d n", n=n)
        action = tf.cast(action, tf.keras.backend.floatx())

        x = tf.concat([agent_loc, action], axis=-2)
        x = tf.cast(agent_loc, tf.keras.backend.floatx())
        x = E.rearrange(x, "b d n -> (b n) d")
        x = self.transition(x)
        x = E.rearrange(x, "(b n) d g -> b d n g", n=n)
        if self.log_space:
            x = tf.nn.log_softmax(x, axis=-1)
        else:
            x = tf.nn.softmax(x, axis=-1)
            x = tf.clip_by_value(x, 1e-6, 1 - 1e-6)
            x = tf.math.log(x)
        return x
