import einops as E
import tensorflow as tf

from keras.layers import *


class NeuralLocationTransition(tf.Module):
    def __init__(self, n_objects):
        super().__init__()
        self.n_objects = n_objects

        self.move_network = tf.keras.Sequential()
        self.move_network.add(Dense(64, activation="relu"))
        self.move_network.add(Dense(32, activation="relu"))
        self.move_network.add(Dense(self.n_objects * 2))
        self.move_network.add(Reshape((self.n_objects, 2)))

    def __call__(
        self,
        locations,
        types,
        held,
        close,
        touching,
        overshoot,
        reset_loc,
        training=False,
        mask=None,
    ):
        n = locations.shape[3]

        locations = E.rearrange(locations, "b o dim n -> (b n) (o dim)")
        types = types[..., 0, :]
        types = tf.one_hot(types, depth=3, dtype=locations.dtype)
        types = E.rearrange(types, "b o n t -> (b n) (o t)")
        held = tf.one_hot(held, depth=self.n_objects, dtype=locations.dtype)
        held = E.repeat(held, "b o -> (b n) o", n=n)
        close = E.rearrange(close, "b o d n -> (b n) (o d)")
        close = tf.cast(close, locations.dtype)
        touching = E.rearrange(touching, "b o d n -> (b n) (o d)")
        touching = tf.cast(touching, locations.dtype)
        overshoot = E.rearrange(overshoot, "b o d n -> (b n) (o d)")
        overshoot = tf.cast(overshoot, locations.dtype)
        reset_loc = E.rearrange(reset_loc, "b o dim n -> (b n) (o dim)")

        x = tf.concat(
            [locations, types, held, close, touching, overshoot, reset_loc], axis=-1
        )
        x = self.move_network(x)
        x = E.rearrange(x, "(b n) o dim -> b o dim n", n=n)
        return x


class SimpleNeuralLocationTransition(tf.Module):
    def __init__(self, n_objects):
        super().__init__()
        self.n_objects = n_objects

        self.move_network = tf.keras.Sequential()
        self.move_network.add(Dense(64, activation="relu"))
        self.move_network.add(Dense(32, activation="relu"))
        self.move_network.add(Dense(self.n_objects * 2))
        self.move_network.add(Reshape((self.n_objects, 2)))

    def __call__(
        self,
        locations,
        types,
        held,
        reset_loc,
        training=False,
        mask=None,
    ):
        n = locations.shape[3]

        locations = E.rearrange(locations, "b o dim n -> (b n) (o dim)")
        types = types[..., 0, :]
        types = tf.one_hot(types, depth=3, dtype=locations.dtype)
        types = E.rearrange(types, "b o n t -> (b n) (o t)")
        held = tf.one_hot(held, depth=self.n_objects, dtype=locations.dtype)
        held = E.repeat(held, "b o -> (b n) o", n=n)
        reset_loc = E.rearrange(reset_loc, "b o dim n -> (b n) (o dim)")

        x = tf.concat([locations, types, held, reset_loc], axis=-1)
        x = self.move_network(x)
        x = E.rearrange(x, "(b n) o dim -> b o dim n", n=n)
        return x
