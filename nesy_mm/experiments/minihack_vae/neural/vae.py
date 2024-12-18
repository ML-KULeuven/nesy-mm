import einops as E
import tensorflow as tf
import tensorflow_probability as tfp

from keras.layers import *


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim, image_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape

        self.model = tf.keras.Sequential()
        self.model.add(Conv2D(16, (5, 5), activation="relu", padding="same"))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Conv2D(32, (5, 5), activation="relu", padding="same"))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(latent_dim * 2))
        self.model.add(Reshape((latent_dim, 2)))

    def __call__(self, inputs, training=False, mask=None):
        x = self.model(inputs)
        mu = x[..., 0]
        sigma = tf.nn.softplus(x[..., 1])
        return mu, sigma

    def call(self, inputs, training=False, mask=None):
        return self.__call__(inputs, training=training, mask=mask)


class Decoder(tf.keras.Model):
    def __init__(self, grid_size, image_shape, dropout):
        super().__init__()
        self.grid_size = grid_size + 2
        self.image_shape = list(image_shape)
        self.w = image_shape[0]
        self.h = image_shape[1]
        self.c = image_shape[2]

        """ Deconvolution decoder """
        latent_channels = 64
        init_size = image_shape[0] // (2**4)

        self.model = tf.keras.Sequential()
        self.model.add(Dense(init_size**2 * 3, activation="relu"))
        self.model.add(Reshape((init_size, init_size, 3)))
        for i in range(3, 0, -1):
            self.model.add(
                Conv2DTranspose(
                    latent_channels * (2**i),
                    (5, 5),
                    activation="relu",
                    padding="same",
                    strides=(2, 2),
                )
            )
            self.model.add(Dropout(dropout))
        self.model.add(
            Conv2DTranspose(
                3, (5, 5), activation="sigmoid", padding="same", strides=(2, 2)
            )
        )

    def __call__(self, inputs, training=False, mask=None):
        agent = inputs[0]
        agent = tf.one_hot(agent, self.grid_size, axis=-1)  # b, d, n, domain
        agent = E.rearrange(agent, "b d n domain -> b n (d domain)")  # b, n, domain * d

        background = inputs[1]
        background = E.rearrange(background, "b d n -> b n d")

        n = background.shape[1]

        x = tf.concat([background, agent], -1)  # b, n, domain * d + d
        x = E.rearrange(x, "b n d -> (b n) d")
        x = self.model(x)
        x = E.rearrange(x, "(b n) h w c -> b n h w c", n=n)
        return x

    def call(self, inputs, training=False, mask=None):
        return self.__call__(inputs, training=training, mask=mask)
