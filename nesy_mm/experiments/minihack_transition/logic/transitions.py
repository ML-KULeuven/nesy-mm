import einops as E
import numpy as np
import tensorflow as tf


EPS = tf.keras.backend.epsilon()


class MoveTransition(tf.Module):
    def __init__(self):
        super().__init__()

        self.x_add = np.zeros([1, 2, 1])
        self.x_add[:, 0, :] = 1
        self.x_add = tf.constant(self.x_add, dtype=tf.int32)

        self.y_add = np.zeros([1, 2, 1])
        self.y_add[:, 1, :] = 1
        self.y_add = tf.constant(self.y_add, dtype=tf.int32)

    def __call__(self, inputs, training=False, mask=None):
        samples = inputs[0]
        action = inputs[1]
        if action.shape[-1] == 8:
            action = tf.argmax(action, axis=-1)
            action = tf.expand_dims(action, axis=-2)
        elif action.shape[-1] == 4:
            action = tf.argmax(action, axis=-1)
            action = E.rearrange(action, "b -> b 1 1")
        else:
            action = E.rearrange(action, "... n -> ... 1 n")
        grid_size = inputs[2]

        # 0:N, 1: E, 2:S, 3:W, 4: NE, 5:SE, 6:SW, 7:NW
        # N: left, E: down, S: right, W: up, NE: down-left, SE: down-right, SW: up-right, NW: up-left

        up_samples = tf.where(action == 3, tf.maximum(samples - self.y_add, 1), samples)
        down_samples = tf.where(
            action == 1, tf.minimum(samples + self.y_add, grid_size - 2), samples
        )
        left_samples = tf.where(
            action == 0, tf.maximum(samples - self.x_add, 1), samples
        )
        right_samples = tf.where(
            action == 2, tf.minimum(samples + self.x_add, grid_size - 2), samples
        )
        up_left_samples = tf.where(
            action == 7,
            tf.maximum(samples - self.y_add - self.x_add, 1),
            samples,
        )
        up_right_samples = tf.where(
            action == 6,
            tf.minimum(tf.maximum(samples - self.y_add, 1) + self.x_add, grid_size - 2),
            samples,
        )
        down_left_samples = tf.where(
            action == 4,
            tf.maximum(tf.minimum(samples + self.y_add, grid_size - 2) - self.x_add, 1),
            samples,
        )
        down_right_samples = tf.where(
            action == 5,
            tf.minimum(samples + self.y_add + self.x_add, grid_size - 2),
            samples,
        )

        deterministic_probs = up_samples
        deterministic_probs += down_samples
        deterministic_probs += left_samples
        deterministic_probs += right_samples
        deterministic_probs += up_left_samples
        deterministic_probs += up_right_samples
        deterministic_probs += down_left_samples
        deterministic_probs += down_right_samples
        deterministic_probs -= 7 * samples
        deterministic_probs = tf.one_hot(
            deterministic_probs,
            grid_size,
            axis=-1,
        )

        next_logits = tf.where(deterministic_probs == 1, 0.0, tf.math.log(EPS))
        return next_logits
