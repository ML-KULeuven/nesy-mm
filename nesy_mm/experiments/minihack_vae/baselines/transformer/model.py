import os
import tensorflow as tf
import einops as E
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import *


class BaseAttention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):

    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x, key=context, value=context, return_attention_scores=True
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class CausalSelfAttention(BaseAttention):

    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class ImageEmbedding(tf.keras.Model):

    def __init__(self, grid_size, image_shape, latent_dim=2):
        super().__init__()
        self.grid_size = grid_size + 2
        self.latent_dim = latent_dim

        self.encoder = Encoder(latent_dim)
        self.classifier = AgentClassifier(grid_size, constraint=False)

    def positional_encoding(self, length, depth):
        depth = depth / 2

        positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

        angle_rates = 1 / (10000**depths)  # (1, depth)
        angle_rads = positions * angle_rates  # (pos, depth)

        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, images, training=False, mask=None):
        t = images.shape[1]

        images = E.rearrange(images, "b t h w c -> (b t) h w c")

        x1 = self.encoder(images)
        x1 = tf.stack(x1, axis=-1)
        x1 = E.rearrange(x1, "(b t) d l -> b t (d l)", t=t, l=self.latent_dim)

        x2 = self.classifier(images)
        x2 = E.rearrange(x2, "(b t) d g -> b t (d g)", t=t, g=self.grid_size)

        x = tf.concat([x1, x2], axis=-1)
        x *= tf.math.sqrt(tf.cast(x.shape[-1], tf.float32))
        x += self.positional_encoding(x.shape[-2], x.shape[-1])[tf.newaxis, :, :]
        return x


class VAEZeroEncoder(tf.keras.Model):

    def __init__(self, latent_dim, image_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)

    def call(self, image, training=False, mask=None):
        x = self.encoder(image)
        x = tf.stack(x, axis=-1)

        mu, sigma = x[..., 0], x[..., 1]
        z = mu + sigma * tf.random.normal(sigma.shape)

        return z, x


class Encoder(tf.keras.Model):

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.model = tf.keras.Sequential()
        self.model.add(
            tf.keras.layers.Conv2D(16, (5, 5), activation="relu", padding="same")
        )
        self.model.add(tf.keras.layers.MaxPool2D((2, 2)))
        self.model.add(
            tf.keras.layers.Conv2D(32, (5, 5), activation="relu", padding="same")
        )
        self.model.add(tf.keras.layers.MaxPool2D((2, 2)))
        # self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        # self.model.add(tf.keras.layers.MaxPool2D((2, 2)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(latent_dim * 2))
        self.model.add(tf.keras.layers.Reshape((latent_dim, 2)))

    def call(self, inputs, training=False, mask=None):
        x = self.model(inputs)
        return [x[:, :, 0], tf.nn.softplus(x[:, :, 1])]


class Decoder(tf.keras.Model):

    def __init__(self, image_shape, dropout=0.4):
        super().__init__()
        self.image_shape = list(image_shape)

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

    def call(self, z, training=False, mask=None):
        x = self.model(z)
        return x


class Classifier(tf.keras.Model):

    def __init__(self, grid_size, constraint=True):
        super().__init__()
        self.grid_size = grid_size + 2
        self.constraint_present = constraint

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(32, activation="relu"))
        # self.model.add(tf.keras.layers.Dense(8, activation="relu"))
        self.model.add(tf.keras.layers.Dense(2 * self.grid_size, activation="linear"))
        self.model.add(tf.keras.layers.Reshape((2, self.grid_size)))

        self.constraint = np.ones([1, 2, self.grid_size])
        self.constraint[:, :, 0] = -np.inf
        self.constraint[:, :, self.grid_size - 1] = -np.inf
        self.bias = np.ones([1, 2, self.grid_size])
        self.bias[:, :, 0] = 0
        self.bias[:, :, self.grid_size - 1] = 0

        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, inputs, training=False, mask=None):
        x = self.model(inputs)
        if self.constraint_present:
            x = x * self.bias + self.constraint
        return self.softmax(x)


class AgentClassifier(tf.keras.Model):

    def __init__(self, grid_size, constraint=True, log_space=False):
        super().__init__()
        self.grid_size = grid_size + 2
        self.constraint_present = constraint
        self.log_space = log_space

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(16, 5, activation="relu"))
        self.model.add(tf.keras.layers.MaxPool2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(32, 5, activation="relu"))
        self.model.add(tf.keras.layers.MaxPool2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(64, 5, activation="relu"))
        self.model.add(tf.keras.layers.MaxPool2D((2, 2)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(32, activation="relu"))
        # self.model.add(tf.keras.layers.Dense(8, activation="relu"))
        self.model.add(tf.keras.layers.Dense(2 * self.grid_size, activation="linear"))
        self.model.add(tf.keras.layers.Reshape((2, self.grid_size)))

        self.constraint = np.ones([1, 2, self.grid_size])
        self.constraint[:, :, 0] = -np.inf
        self.constraint[:, :, self.grid_size - 1] = -np.inf
        self.bias = np.ones([1, 2, self.grid_size])
        self.bias[:, :, 0] = 0
        self.bias[:, :, self.grid_size - 1] = 0

        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, inputs):
        x = self.model(inputs)
        if self.constraint_present:
            x = x * self.bias + self.constraint
        # x = self.softmax(x)
        return x


class Bumblebee(tf.keras.Model):

    def __init__(
        self,
        grid_size,
        horizon,
        image_shape,
        downsample,
        num_heads=8,
        key_dim=64,
        dropout=0.1,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout
        self.image_shape = image_shape
        self.horizon = horizon
        self.downsample = downsample

        # Latent space of 9 as that is the size of the embedding used by the transformer encoder

        if grid_size == 10:
            self.zero_vae = VAEZeroEncoder(14, image_shape)
        else:
            self.zero_vae = VAEZeroEncoder(9, image_shape)

        self.embedding = ImageEmbedding(grid_size, image_shape)
        self.dropout = Dropout(dropout)

        self.causal_attn = CausalSelfAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout
        )
        self.cross_attn = CrossAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout
        )
        self.decoder = Decoder(self.image_shape)
        self.classifier = Classifier(grid_size, constraint=False)

    def call(self, images, actions, training=False, mask=None):
        image_zero = images[:, 0, ...]
        image_t = images[:, 1:, ...]

        t = image_t.shape[1]

        z_zero, posterior_zero = self.zero_vae(image_zero)
        posterior_zero = tf.expand_dims(posterior_zero, axis=1)
        generation_zero = self.decoder(z_zero)
        generation_zero = tf.expand_dims(generation_zero, axis=1)

        x = self.embedding(image_t)
        x = self.dropout(x)
        x = self.causal_attn(x)

        posterior_context = E.rearrange(x, "b t (d l) -> b t d l", l=2)

        mu = posterior_context[..., 0]
        sigma = tf.nn.softplus(posterior_context[..., 1])
        z = mu + sigma * tf.random.normal(sigma.shape)

        posteriors = tf.concat([posterior_zero, posterior_context], axis=1)

        actions = tf.cast(actions, tf.float32)
        x = self.cross_attn(z, actions)

        x_final = x[:, -1, ...]
        final_mario = self.classifier(x_final)

        x = E.rearrange(x, "b t s -> (b t) s")
        generations = self.decoder(x)
        generations = E.rearrange(generations, "(b t) h w c -> b t h w c", t=t)
        generations = tf.concat([generation_zero, generations], axis=1)

        return generations, posteriors, final_mario

    def generate_next(self, input_sequence, actions):
        t = input_sequence.shape[1]

        x = self.embedding(input_sequence)
        x = self.dropout(x)
        x = self.causal_attn(x)

        posterior_context = E.rearrange(x, "b t (d l) -> b t d l", l=2)

        mu = posterior_context[..., 0]
        sigma = tf.nn.softplus(posterior_context[..., 1])
        z = mu + sigma * tf.random.normal(sigma.shape)

        x = self.cross_attn(z, actions)

        x = E.rearrange(x, "b t s -> (b t) s")
        generations = self.decoder(x)
        generations = E.rearrange(generations, "(b t) h w c -> b t h w c", t=t)

        return generations[:, -1, ...]

    def generate(self, actions, training=False, mask=None):
        latent_size = 9 if self.grid_size == 5 else 14
        prior_sample = tf.random.normal((actions.shape[0], latent_size))

        start_image = self.decoder(prior_sample)

        output_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        output_array = output_array.write(0, start_image)

        horizon = actions.shape[1]
        for i in tf.range(horizon):
            output = output_array.stack()
            output = E.rearrange(output, "t b h w c -> b t h w c")
            next_generation = self.generate_next(output, actions)
            output_array = output_array.write(i + 1, next_generation)

        generated_sequence = output_array.stack()
        generated_sequence = E.rearrange(generated_sequence, "t b h w c -> b t h w c")
        return generated_sequence

    def plot_sequence(self, sequence, epoch=0, horizon=10, seed=0):
        n = len(sequence)
        fig, axs = plt.subplots(1, n, figsize=(n * 5, 5))
        for i, generation in enumerate(sequence):
            ax = fig.add_subplot(1, n, i + 1)
            ax.imshow(generation[0, ...])
            ax.axis("off")

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        folder = "nesy_mm/plots"
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = f"transformer_generation_epoch_{epoch}_horizon{horizon}_grid_size{self.grid_size}_seed{seed}.png"

        plt.savefig(
            os.path.join(folder, filename),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
