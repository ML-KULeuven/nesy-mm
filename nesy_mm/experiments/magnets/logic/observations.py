import tensorflow as tf

from nesy_mm.src.logic.comparisons import EqualTo


class Consistency(tf.Module):
    def __init__(self, n_objects, log_space=False):
        super().__init__()
        self.n_objects = n_objects
        self.log_space = log_space
        self.equal_to = EqualTo(log_space=log_space)

    def __call__(self, planned_location, location_t, training=False, mask=None):
        return self.equal_to(planned_location, location_t)
