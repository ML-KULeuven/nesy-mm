import numpy as np
import einops as E
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd

from nesy_mm.experiments.minihack_vae.probabilistic.transitions import AgentTransition
from nesy_mm.experiments.minihack_vae.neural.vae import Encoder, Decoder
from nesy_mm.experiments.minihack_vae.neural.classifier import AgentClassifier

from nesy_mm.src.probabilistic.combinations import CombinationConstructor
from nesy_mm.src.probabilistic.resampling import FiniteResampler


class MinihackVAEModel(tf.Module):
    def __init__(
        self,
        grid_size,
        n_samples,
        latent_dim,
        image_shape,
        dropout,
        horizon,
        downsample=1,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.n_samples = n_samples
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.horizon = horizon
        self.downsample = downsample

        self.encoder = Encoder(latent_dim, image_shape)
        self.decoder = Decoder(grid_size, image_shape, dropout)
        self.classifier = AgentClassifier(grid_size, image_shape)
        self.transition = AgentTransition(grid_size)

    def __call__(self, inputs, training=False, mask=None):
        images = inputs[0]
        actions = inputs[1]

        mu, sigma = self.encoder(images[:, 0])
        background = tfd.Normal(mu, sigma).sample(self.n_samples)
        background = E.rearrange(background, "n b d -> b d n")

        logits_0 = self.classifier(images[:, 0])
        agent_0 = tfd.Categorical(logits=logits_0).sample(self.n_samples)
        agent_0 = E.rearrange(agent_0, "n b d -> b d n")
        logits_0 = tf.gather(logits_0, agent_0, axis=-1, batch_dims=2)
        logits_0 = tf.reduce_sum(logits_0, axis=1)

        generation_0 = self.decoder([agent_0, background])
        generations = [generation_0]
        logits = [logits_0]
        for t in range(self.horizon - 1):
            logits_t = self.transition([agent_0, actions[:, t]])
            agent_t = tfd.Categorical(logits=logits_t).sample(1)[0]
            logits_t = tf.gather(logits_t, agent_t, axis=-1, batch_dims=3)
            logits_t = tf.reduce_sum(logits_t, axis=1)
            generation_t = self.decoder([agent_t, background])
            agent_0 = agent_t

            generations.append(generation_t)
            logits.append(logits_t)

        generations = tf.stack(generations, axis=1)
        logits = tf.stack(logits, axis=1)
        logits = tf.reduce_sum(logits, axis=1)
        return generations, [mu, sigma], agent_0, logits

    def generate(self, actions, n_samples=1, starting_loc=None):
        # if no starting location is provided, sample one
        if starting_loc is None:
            loc = tf.random.uniform(
                (1, 2, n_samples), 0, self.grid_size + 2, dtype=tf.int32  # b, d, n
            )
        else:
            loc = E.repeat(starting_loc, "d -> 1 d n", n=n_samples)

        actions = E.rearrange(actions, "h -> 1 h")

        mu = tf.zeros((n_samples, self.latent_dim))
        sigma = tf.ones((n_samples, self.latent_dim))
        background = tfd.Normal(mu, sigma).sample()
        background = E.rearrange(background, "n d -> 1 d n")

        generation_0 = self.decoder([loc, background])
        generations = [generation_0]
        for t in range(actions.shape[1]):
            logits_t = self.transition([loc, actions[:, t]])
            agent_t = tfd.Categorical(logits=logits_t).sample(1)[0]
            generation_t = self.decoder([agent_t, background])
            loc = agent_t

            generations.append(generation_t)

        return tf.stack(generations, axis=1)

    def generate_with_constraint(self, actions):
        """
        Generate a sequence of images with a constraint on the agent's location.
        Specifically, the agent must avoid the last two columns of the grid.
        """
        combination_constructor = CombinationConstructor()
        resampler = FiniteResampler()

        loc = tf.constant([3, 3], dtype=tf.int32)
        loc = E.repeat(loc, "d -> 1 d n", n=1)

        actions = E.rearrange(actions, "h -> 1 h")

        mu = tf.zeros((1, self.latent_dim))
        sigma = tf.ones((1, self.latent_dim))
        background = tfd.Normal(mu, sigma).sample()
        background = E.rearrange(background, "n d -> 1 d n")

        generation_0 = self.decoder([loc, background])
        generations = [generation_0]
        for t in range(actions.shape[1]):
            logits_t = self.transition([loc, actions[:, t]], mask="constraint")

            # Ensure agent does not move to the last two columns
            combinations, combinations_weights = combination_constructor(
                [logits_t], [list(range(self.grid_size + 2))]
            )  # [(2 49)], (1 n 49)

            combination_observations = self.in_last_two_columns(combinations[0])
            combination_observations = E.rearrange(
                combination_observations, "combs d -> 1 1 combs d"
            )

            observation = tf.zeros(
                (1, 1, combination_observations.shape[-2]), dtype=tf.int32
            )  # because we want to forbid the last two columns

            agent_t, _ = resampler(
                "categorical",
                combination_observations,
                observation,
                combinations_weights,
                combinations,
            )
            agent_t = agent_t[0]

            generation_t = self.decoder([agent_t, background])
            loc = agent_t

            generations.append(generation_t)

        return tf.stack(generations, axis=1)

    def in_last_two_columns(self, agent_loc):
        in_columns = tf.where(agent_loc[1, :] > self.grid_size - 2, 0.0, -np.inf)
        not_in_columns = tf.where(agent_loc[1, :] <= self.grid_size - 2, 0.0, -np.inf)
        in_columns = tf.stack([not_in_columns, in_columns], axis=-1)
        return in_columns

    def save_weights(self, path):
        self.encoder.save_weights(f"{path}.encoder")
        self.decoder.save_weights(f"{path}.decoder")
        self.classifier.save_weights(f"{path}.classifier")

    def load_weights(self, path):
        self.encoder.load_weights(f"{path}.encoder")
        self.decoder.load_weights(f"{path}.decoder")
        self.classifier.load_weights(f"{path}.classifier")
