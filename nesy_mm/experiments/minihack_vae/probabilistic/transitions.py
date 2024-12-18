import einops as E
import numpy as np
import tensorflow as tf


class AgentTransition(tf.keras.Model):
    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size + 2

        self.x_add = np.zeros([1, 2, 1])
        self.x_add[:, 0, :] = 1
        self.x_add = tf.constant(self.x_add, dtype=tf.int32)

        self.y_add = np.zeros([1, 2, 1])
        self.y_add[:, 1, :] = 1
        self.y_add = tf.constant(self.y_add, dtype=tf.int32)

    def __call__(self, inputs, training=False, mask=None):
        samples = inputs[0]
        action = inputs[1]
        if action.shape[-1] == 4:
            action = tf.argmax(action, axis=-1)
        action = E.rearrange(action, "b -> b 1 1")

        # 0:N, 1: E, 2:S, 3:W, 4: NE, 5:SE, 6:SW, 7:NW
        # N: left, E: down, S: right, W: up, NE: down-left, SE: down-right, SW: up-right, NW: up-left

        up_samples = tf.where(action == 3, tf.maximum(samples - self.y_add, 1), samples)
        down_samples = tf.where(
            action == 1, tf.minimum(samples + self.y_add, self.grid_size - 2), samples
        )
        left_samples = tf.where(
            action == 0, tf.maximum(samples - self.x_add, 1), samples
        )
        right_samples = tf.where(
            action == 2, tf.minimum(samples + self.x_add, self.grid_size - 2), samples
        )

        deterministic_probs = up_samples
        deterministic_probs += down_samples
        deterministic_probs += left_samples
        deterministic_probs += right_samples
        deterministic_probs -= 3 * samples
        deterministic_probs = tf.one_hot(
            deterministic_probs,
            self.grid_size,
            axis=-1,
        )

        next_logits = tf.where(deterministic_probs == 1, 0.0, -np.inf)
        if mask == "constraint":
            previous_loc = tf.one_hot(samples, self.grid_size, axis=-1)
            next_probs = (
                tf.where(deterministic_probs == 1, 1.0 - 1e-7, 0) + 1e-7 * previous_loc
            )
            next_logits = tf.math.log(next_probs)
        return next_logits

    def call(self, inputs, training=False, mask=None):
        return self.__call__(inputs, training=training, mask=mask)
