import numpy as np
import tensorflow as tf


EPS = tf.keras.backend.epsilon()


class Hit(tf.keras.Model):
    """
    This class is responsible for handling the logic of the hit observation.
    It takes in input the agent location and the enemy location and returns the
    probability of the enemy hitting the agent. If the enemy hits the agent, the
    enemy must be in one of the eight adjacent squares, and it hits with
    probability 1/8. Otherwise, the enemy misses the agent.
    """

    def __init__(self):
        super().__init__()
        self.hit_chance = tf.Variable(
            0.0, trainable=True, dtype=tf.keras.backend.floatx()
        )

    def __call__(self, agent_loc, enemy_loc, training=False, mask=None):
        """
        This function takes in input the agent location and the enemy location
        and returns the probability of the enemy hitting the agent.
        """
        # Compute the difference between the agent and enemy locations
        diff = tf.abs(agent_loc - enemy_loc)
        diff = tf.reduce_sum(diff, axis=-2)

        hit_chance = tf.math.log_sigmoid(self.hit_chance)

        # If the enemy is in one of the eight adjacent squares, the enemy hits
        hit = tf.where(diff <= 1, hit_chance, tf.math.log(EPS))
        return hit
