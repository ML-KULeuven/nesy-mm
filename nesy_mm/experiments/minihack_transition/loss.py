import einops as E
import tensorflow as tf

from typing import Any
from nesy_mm.src.logic.comparisons import EqualTo


EPS = tf.keras.backend.epsilon()


class DeadOrNot(tf.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, targets, inputs, training=False, mask=None):
        deadornot = targets

        deadornot_hat = inputs

        indicator = tf.where(deadornot == deadornot_hat, 1.0, EPS)
        indicator = tf.cast(indicator, tf.keras.backend.floatx())
        return indicator
