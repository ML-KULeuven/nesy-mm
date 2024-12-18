import einops as E
import tensorflow as tf

from abc import abstractmethod
from nesy_mm.src.logic.comparisons import SmallerThan


EPS = tf.keras.backend.epsilon()


class LocationProperty(tf.Module):
    def __init__(self, n_objects, log_space):
        super().__init__()
        self.n_objects = n_objects
        self.log_space = log_space

    @abstractmethod
    def property(self, location_i, location_j):
        pass

    def __call__(self, locations, training=False, mask=None):
        log_p = []
        for i in range(self.n_objects):
            location_i = locations[..., i, :, :]
            log_p_i = []
            for j in range(self.n_objects):
                if i != j:
                    location_j = locations[..., j, :, :]
                    log_p_ij = self.property(location_i, location_j)
                    log_p_i.append(log_p_ij)

            log_p_i = tf.stack(log_p_i, axis=-2)
            log_p.append(log_p_i)

        log_p = tf.stack(log_p, axis=-3)
        return log_p


class Close(LocationProperty):
    """
    This class is used to check if two objects are close to each others.
    They are close if they are within a distance of 3 * step_size (i.e. interaction radius).
    """

    def __init__(self, n_objects, log_space):
        super().__init__(n_objects, log_space)
        self.step_size = 0.1
        self.smaller_than = SmallerThan(log_space=log_space)

    def property(self, location_i, location_j):
        distance = tf.norm(location_i - location_j, axis=-2)
        interaction_threshold = tf.constant(3 * self.step_size, dtype=distance.dtype)

        return self.smaller_than(distance, interaction_threshold)


class Touching(LocationProperty):
    def __init__(self, n_objects, log_space):
        super().__init__(n_objects, log_space)
        self.step_size = 0.1
        self.object_radius = 0.02
        self.noise = 1e-2
        self.smaller_than = SmallerThan(log_space=log_space)

    def property(self, location_i, location_j):
        distance = tf.norm(location_i - location_j, axis=-2)
        touching_distance = tf.constant(
            self.object_radius * 2 + self.noise, dtype=distance.dtype
        )

        return self.smaller_than(distance, touching_distance)


class Overshoot(LocationProperty):
    def __init__(self, n_objects, log_space):
        super().__init__(n_objects, log_space=log_space)
        self.step_size = 0.1
        self.object_radius = 0.02
        self.magnetic_constant = 0.05
        self.smaller_than = SmallerThan(log_space=log_space)

    def property(self, location_i, location_j):
        distance = tf.norm(location_i - location_j, axis=-2, keepdims=True)
        magnetic_force = self.magnetic_constant / (distance**2 + EPS)

        direction = location_j - location_i
        direction_x = direction[..., 0, :]
        direction_y = direction[..., 1, :]
        direction_normalised = direction / (distance + EPS)
        direction_angle = tf.atan2(direction_x, direction_y)
        direction_diameter = tf.stack(
            [
                tf.sin(direction_angle) * self.object_radius * 2,
                tf.cos(direction_angle) * self.object_radius * 2,
            ],
            axis=-2,
        )

        movement = direction_normalised * magnetic_force * self.step_size
        movement = tf.norm(movement, axis=-2)

        safe_distance = direction - direction_diameter
        safe_distance = tf.norm(safe_distance, axis=-2)
        return self.smaller_than(safe_distance, movement)
