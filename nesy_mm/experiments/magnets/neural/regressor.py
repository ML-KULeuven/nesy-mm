import einops as E
import tensorflow as tf

from keras.layers import *


class Regressor(tf.keras.Model):

    def __init__(self, n_objects):
        super().__init__()
        self.n_objects = n_objects

        self.model = tf.keras.Sequential()
        self.model.add(Conv2D(32, (5, 5), activation="relu"))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (5, 5), activation="relu"))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (5, 5), activation="relu"))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(4 * n_objects))
        self.model.add(Reshape((n_objects, 2, 2)))

    def call(self, image, training=False, mask=None):
        x = self.model(image)

        mu = x[..., 0]
        mu = tf.nn.sigmoid(mu)
        sigma = tf.nn.softplus(x[..., 1])
        return mu, sigma
