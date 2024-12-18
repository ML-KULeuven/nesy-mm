import einops as E
import tensorflow as tf

from keras.layers import *
from nesy_mm.experiments.magnets.data.generator import ObjectType


EPS = tf.keras.backend.epsilon()


class LocationTransition(tf.Module):
    def __init__(self, n_objects):
        super().__init__()
        self.n_objects = n_objects
        self.step_size = 0.1
        self.object_radius = 0.02
        self.magnetic_constant = 0.05

    def check_interaction(self, type_i, type_j, close_ij):
        magnet_to_magnet = tf.logical_and(
            type_i == ObjectType.MAGNET, type_j == ObjectType.MAGNET
        )
        magnet_to_ferromagnetic = tf.logical_and(
            type_i == ObjectType.MAGNET, type_j == ObjectType.FERROMAGNETIC
        )
        ferromagnetic_to_magnet = tf.logical_and(
            type_i == ObjectType.FERROMAGNETIC, type_j == ObjectType.MAGNET
        )

        types_can_interact = tf.logical_or(
            magnet_to_magnet,
            tf.logical_or(magnet_to_ferromagnetic, ferromagnetic_to_magnet),
        )

        interaction = tf.logical_and(types_can_interact, close_ij == 1)
        return interaction

    def move_location(self, location_i, location_j, interaction):
        distance = tf.norm(location_i - location_j, axis=-2, keepdims=True)
        magnetic_force = self.magnetic_constant / (distance**2 + EPS)

        direction = location_j - location_i
        direction_norm = direction / (distance + EPS)

        movement = direction_norm * magnetic_force * self.step_size

        next_object_location = location_i + movement
        next_object_location = tf.where(interaction, next_object_location, location_i)
        return next_object_location

    def check_valid_movement(self, interaction, overshoot, held_i):
        real_interaction = tf.logical_and(interaction, tf.logical_not(held_i))
        no_overshoot = tf.logical_not(overshoot == 1)

        valid_movement = tf.logical_and(real_interaction, no_overshoot)
        return valid_movement

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
        next_locations = []
        for i in range(self.n_objects):
            held_i = held == i
            held_i = E.rearrange(held_i, "b -> b 1 1")

            type_i = types[..., i, 0, :]
            type_i = tf.expand_dims(type_i, -2)
            location_i = locations[..., i, :, :]

            # Each object interacts with each other object except itself
            count_j = 0
            for j in range(self.n_objects):
                if i != j:
                    close_ij = close[..., i, count_j, :]
                    close_ij = tf.expand_dims(close_ij, -2)

                    touching_ij = touching[..., i, count_j, :]
                    touching_ij = tf.expand_dims(touching_ij, -2)

                    type_j = types[..., j, 0, :]
                    type_j = tf.expand_dims(type_j, -2)
                    location_j = locations[..., j, :, :]

                    interaction = self.check_interaction(type_i, type_j, close_ij)
                    valid_movement = self.check_valid_movement(
                        interaction, overshoot[..., i : i + 1, count_j, :], held_i
                    )

                    next_location_i = self.move_location(
                        location_i, location_j, interaction
                    )
                    next_location_i = tf.where(
                        valid_movement, next_location_i, location_i
                    )

                    count_j += 1

            next_locations.append(next_location_i)
        next_locations = tf.stack(next_locations, axis=-3)

        any_are_touching = tf.cast(touching, tf.bool)
        any_are_touching = tf.reduce_any(any_are_touching, axis=-2, keepdims=True)
        any_are_touching = tf.reduce_any(any_are_touching, axis=-3, keepdims=True)
        next_locations = tf.where(any_are_touching, reset_loc, next_locations)
        return next_locations
