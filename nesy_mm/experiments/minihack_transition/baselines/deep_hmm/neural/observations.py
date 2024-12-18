import einops as E
import tensorflow as tf

from keras.layers import *

EPS = tf.keras.backend.epsilon()


class NeuralHit(tf.keras.Model):
    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.hit_network = tf.keras.Sequential()
        self.hit_network.add(Dense(64, activation="relu"))
        self.hit_network.add(Dense(32, activation="relu"))
        self.hit_network.add(Dense(1))

    def __call__(self, agent_loc, enemy_loc, training=False, mask=None):
        """
        This function takes in input the agent location and the enemy location
        and returns the probability of the enemy hitting the agent.
        """
        agent_loc = tf.ones_like(enemy_loc) * agent_loc
        enemy_loc = tf.ones_like(agent_loc) * enemy_loc

        n = agent_loc.shape[-1]
        e = enemy_loc.shape[1]

        x = tf.concat([agent_loc, enemy_loc], axis=-2)
        x = tf.cast(x, tf.keras.backend.floatx())
        x = E.rearrange(x, "b e d n -> (b e n) d")
        x = self.hit_network(x)
        x = E.rearrange(x, "(b e n) d -> b e n d", n=n, e=e)
        x = tf.math.log_sigmoid(x)
        x = x[..., 0]
        return x
