import numpy as np
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd

from keras.layers import *


class AgentClassifier(tf.keras.Model):
    def __init__(self, grid_size, image_shape):
        super().__init__()
        self.grid_size = grid_size + 2
        self.image_shape = image_shape

        self.model = tf.keras.Sequential()
        self.model.add(Conv2D(16, 5, activation="relu"))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Conv2D(32, 5, activation="relu"))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Conv2D(64, 5, activation="relu"))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(2 * self.grid_size, activation="linear"))
        self.model.add(Reshape((2, self.grid_size)))

        self.constraint = np.zeros([1, 2, self.grid_size])
        self.constraint[:, :, 0] = -np.inf
        self.constraint[:, :, self.grid_size - 1] = -np.inf

    def __call__(self, inputs, training=False, mask=None):
        x = self.model(inputs)
        x = x + self.constraint
        x = tf.nn.log_softmax(x, axis=-1)
        return x

    def call(self, inputs, training=False, mask=None):
        return self.__call__(inputs, training=training, mask=mask)
