import einops as E
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd

from keras.layers import *
from nesy_mm.experiments.minihack_transition.logic.transitions import MoveTransition


class EnemyTransition(tf.keras.Model):
    """
    An MLP that models the transition of an enemy. It takes as input the location of
    the enemy and the location of the agent, and outputs the logits of the next
    location of the enemy.
    """

    def __init__(self, grid_size, log_space):
        super().__init__()
        self.grid_size = grid_size + 2
        self.log_space = log_space

        self.move_network = tf.keras.Sequential()
        self.move_network.add(Dense(64, activation="relu"))
        self.move_network.add(Dense(32, activation="relu"))
        self.move_network.add(Dense(self.grid_size * 2))
        self.move_network.add(Reshape((2, self.grid_size)))

    def __call__(self, enemy_loc, agent_loc, training=False, mask=None):
        n = enemy_loc.shape[-1]
        e = enemy_loc.shape[-3]

        x = tf.concat([enemy_loc, agent_loc], axis=-2)
        x = tf.cast(x, tf.keras.backend.floatx())
        x = E.rearrange(x, "b e d n -> (b e n) d")
        x = self.move_network(x)
        x = E.rearrange(x, "(b e n) d g -> b e d n g", e=e, n=n)
        if self.log_space:
            x = tf.nn.log_softmax(x, axis=-1)
        else:
            x = tf.nn.softmax(x, axis=-1)
            x = tf.clip_by_value(x, 1e-6, 1 - 1e-6)
            x = tf.math.log(x)
        return x


class EnemyAction(tf.keras.Model):
    def __init__(self, log_space, relational):
        super().__init__()
        self.log_space = log_space
        self.relational = relational

        self.action_network = tf.keras.Sequential()
        self.action_network.add(Dense(64, activation="relu"))
        self.action_network.add(Dense(32, activation="relu"))
        self.action_network.add(Dense(8))

    def __call__(self, enemy_loc, agent_loc, training=False, mask=None):
        n = enemy_loc.shape[-1]
        e = enemy_loc.shape[-3]

        if self.relational:
            x = enemy_loc - agent_loc
            x = tf.where(x > 0, 1.0, 0.0)
        else:
            x = tf.concat([enemy_loc, agent_loc], axis=-2)
        x = tf.cast(x, tf.keras.backend.floatx())
        x = E.rearrange(x, "b e d n -> (b e n) d")
        x = self.action_network(x)
        x = E.rearrange(x, "(b e n) a -> b e n a", n=n, e=e)
        if self.log_space:
            x = tf.nn.log_softmax(x, axis=-1)
        else:
            x = tf.nn.softmax(x, axis=-1)
            x = tf.clip_by_value(x, 1e-6, 1 - 1e-6)
            x = tf.math.log(x)
        return x
