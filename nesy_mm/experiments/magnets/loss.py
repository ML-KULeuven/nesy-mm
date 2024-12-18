import einops as E
import tensorflow as tf

from typing import Any
from nesy_mm.src.logic.comparisons import EqualTo


EPS = tf.keras.backend.epsilon()


class MagnetLoss(tf.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, targets, inputs, training=False, mask=None):
        types = targets[0]
        types = E.rearrange(types, "... -> ... 1 1")

        types_hat = inputs[0]

        indicator = tf.where(types == types_hat, 1.0, 0.0)
        indicator = E.reduce(indicator, "b ... n -> b n", "prod")
        indicator = tf.where(indicator < EPS, indicator + EPS, indicator)
        indicator = tf.cast(indicator, tf.keras.backend.floatx())
        return indicator


class MagnetSupervisedLoss(MagnetLoss):

    def __call__(self, targets, inputs, training=False, mask=None):
        type_loss = super().__call__(targets, inputs, training, mask)

        locations = targets[1]
        locations = E.rearrange(locations, "... -> ... 1")
        locations = tf.cast(locations, tf.keras.backend.floatx())
        locations_hat = inputs[1]

        mse = tf.math.square(locations - locations_hat)
        mse = E.reduce(mse, "b o d n -> b o n", "mean")
        mse = E.reduce(mse, "b o n -> b n", "sum")
        equals = tf.exp(-mse)

        # equals = EqualTo(log_space=False)
        # equals = equals(locations, locations_hat)
        # equals = E.reduce(equals, "b o d n -> b n", "prod")

        loss = equals * type_loss
        return loss


class LogMagnetSupervisedLoss(tf.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, targets, inputs, training=False, mask=None):
        types = targets[0]
        types = E.rearrange(types, "... -> ... 1 1")

        types_hat = inputs[0]

        indicator = tf.where(types == types_hat, 0.0, tf.math.log(EPS))
        indicator = E.reduce(indicator, "b ... n -> b n", "sum")
        indicator = tf.cast(indicator, tf.keras.backend.floatx())

        locations = targets[1]
        locations = E.rearrange(locations, "... -> ... 1")
        locations = tf.cast(locations, tf.keras.backend.floatx())
        locations_hat = inputs[1]

        mse = tf.math.square(locations - locations_hat)
        mse = E.reduce(mse, "b o d n -> b o n", "mean")
        mse = E.reduce(mse, "b o n -> b n", "sum")
        equals = -mse

        # equals = EqualTo()
        # equals = equals(locations, locations_hat)
        # equals = E.reduce(equals, "b o d n -> b n", "sum")

        loss = equals + indicator
        return -loss
