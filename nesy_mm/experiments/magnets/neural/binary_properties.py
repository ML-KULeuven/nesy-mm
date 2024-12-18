import einops as E
import tensorflow as tf

from keras.layers import *


EPS = tf.keras.backend.epsilon()


class BinaryProperty(tf.keras.Model):

    def __init__(self, n_objects, log_space):
        super().__init__()
        self.n_objects = n_objects
        self.log_space = log_space
        self.activation = "sigmoid" if log_space else "log_sigmoid"

        self.transition_model = tf.keras.Sequential()
        self.transition_model.add(Dense(64, activation="relu"))
        self.transition_model.add(Dense(32, activation="relu"))
        self.transition_model.add(
            Dense(n_objects * (n_objects - 1), activation=self.activation)
        )
        self.transition_model.add(Reshape((n_objects, n_objects - 1)))

    def call(self, locations, training=False, mask=None):
        n = locations.shape[3]

        x = E.rearrange(locations, "b o dim n -> (b n) (o dim)")
        x = self.transition_model(x)
        x = E.rearrange(x, "(b n) o d -> b o d n", n=n)
        return x
