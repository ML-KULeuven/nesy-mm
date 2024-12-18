import einops as E
import tensorflow_probability.python.distributions as tfd

from nesy_mm.src.probabilistic.resampling import InfiniteResampler
from nesy_mm.src.utils import log1mexp

from nesy_mm.experiments.magnets.neural.regressor import Regressor
from nesy_mm.experiments.magnets.neural.binary_properties import BinaryProperty
from nesy_mm.experiments.magnets.neural.transitions import (
    NeuralLocationTransition,
    SimpleNeuralLocationTransition,
)
from nesy_mm.experiments.magnets.probabilistic.transitions import *
from nesy_mm.experiments.magnets.logic.location_properties import *
from nesy_mm.experiments.magnets.logic.observations import Consistency


EPS = tf.keras.backend.epsilon()


class MagnetModel(tf.Module):
    def __init__(self):
        super().__init__()

    def get_bernoulli_parameters(self, parameters):
        if self.log_space:
            logits_1 = parameters
            logits_0 = log1mexp(parameters)
        else:
            logits_1 = tf.clip_by_value(parameters, EPS, 1 - EPS)
            logits_0 = 1 - logits_1
            logits_0 = tf.math.log(logits_0)
            logits_1 = tf.math.log(logits_1)

        logits = tf.stack([logits_0, logits_1], axis=-1)
        return logits

    def sample_bernoulli(self, parameters):
        logits = self.get_bernoulli_parameters(parameters)

        samples = tfd.Categorical(logits=logits)
        samples = samples.sample(1)[0]

        sample_logits = tf.gather(logits, samples, axis=-1, batch_dims=4)
        return samples, sample_logits

    @abstractmethod
    def __call__(self, inputs, training=False, mask=None):
        pass


class LogicMagnetModel(MagnetModel):
    def __init__(self, n_objects, horizon, n_samples, log_space):
        super().__init__()
        self.n_objects = n_objects
        self.horizon = horizon
        self.n_samples = n_samples
        self.log_space = log_space

        self.regressor = Regressor(n_objects)
        self.location_transition = LocationTransition(n_objects)
        self.close = Close(n_objects, log_space)
        self.touching = Touching(n_objects, log_space)
        self.overshoot = Overshoot(n_objects, log_space)
        self.consistency = Consistency(n_objects, log_space)

        self.infinite_resampler = InfiniteResampler()

    def get_bernoulli_parameters(self, parameters):
        if self.log_space:
            logits_1 = parameters
            logits_0 = log1mexp(parameters)
        else:
            logits_1 = tf.clip_by_value(parameters, EPS, 1 - EPS)
            logits_0 = 1 - logits_1
            logits_0 = tf.math.log(logits_0)
            logits_1 = tf.math.log(logits_1)

        logits = tf.stack([logits_0, logits_1], axis=-1)
        return logits

    def sample_bernoulli(self, parameters):
        logits = self.get_bernoulli_parameters(parameters)

        samples = tfd.Categorical(logits=logits)
        samples = samples.sample(1)[0]

        sample_logits = tf.gather(logits, samples, axis=-1, batch_dims=4)
        return samples, sample_logits

    def __call__(self, inputs, training=False, mask=None):
        images = inputs[0]
        images = E.rearrange(images, "b t h w c -> (b t) h w c")

        locations_mu, locations_sigma = self.regressor(images)
        locations_mu = E.rearrange(
            locations_mu, "(b t) o dim -> b t o dim", t=self.horizon
        )
        locations_sigma = E.rearrange(
            locations_sigma, "(b t) o dim -> b t o dim", t=self.horizon
        )

        locations = tfd.Normal(locations_mu, locations_sigma)
        locations = locations.sample(self.n_samples)
        locations = E.rearrange(locations, "n b t o dim -> b t o dim n")

        locations_t = locations[:, 0]
        reset_location = locations[:, 0]

        types_t = tf.zeros([locations_t.shape[0], self.n_objects, 3])
        types_t = tfd.Categorical(logits=types_t)
        types_t = types_t.sample(self.n_samples)
        types_t = E.rearrange(types_t, "n b o -> b o 1 n")

        held = inputs[1]
        observation = tf.ones(
            [locations_t.shape[0], self.n_objects, 2, self.n_samples], dtype=tf.int32
        )

        logits = 0
        for t in range(self.horizon - 1):
            close_t = self.close(locations_t)
            close_t, close_t_logits = self.sample_bernoulli(close_t)
            logits += E.reduce(close_t_logits, "b ... n -> b n", "sum")

            touching_t = self.touching(locations_t)
            touching_t, touching_t_logits = self.sample_bernoulli(touching_t)
            logits += E.reduce(touching_t_logits, "b ... n -> b n", "sum")

            overshoot_t = self.overshoot(locations_t)
            overshoot_t, overshoot_t_logits = self.sample_bernoulli(overshoot_t)
            logits += E.reduce(overshoot_t_logits, "b ... n -> b n", "sum")

            locations_t = self.location_transition(
                locations_t,
                types_t,
                held[:, t],
                close_t,
                touching_t,
                overshoot_t,
                reset_location,
            )

            consistency_t = self.consistency(locations_t, locations[:, t + 1])
            consistency_t = self.get_bernoulli_parameters(consistency_t)

            unconditioned_samples = [locations_t, types_t]
            conditioned_samples, resampling_weights = self.infinite_resampler(
                unconditioned_samples, "categorical", consistency_t, observation
            )
            resampling_weights = E.reduce(resampling_weights, "b ... n -> b n", "sum")

            locations_t = conditioned_samples[0]
            types_t = conditioned_samples[1]
            logits += resampling_weights

        return types_t, locations[:, 0], logits


class DeepHMMMagnetModel(MagnetModel):
    def __init__(self, n_objects, horizon, n_samples):
        super().__init__(n_objects, horizon, n_samples)

        self.location_transition = NeuralLocationTransition(n_objects)
        self.close = BinaryProperty(n_objects)
        self.touching = BinaryProperty(n_objects)
        self.overshoot = BinaryProperty(n_objects)


class SimpleDeepHMMModel(MagnetModel):
    def __init__(self, n_objects, horizon, n_samples, log_space):
        super().__init__()
        self.n_objects = n_objects
        self.horizon = horizon
        self.n_samples = n_samples
        self.log_space = log_space

        self.regressor = Regressor(n_objects)
        self.location_transition = SimpleNeuralLocationTransition(n_objects)
        self.consistency = Consistency(n_objects, log_space)

        self.infinite_resampler = InfiniteResampler()

    def __call__(self, inputs):
        images = inputs[0]
        images = E.rearrange(images, "b t h w c -> (b t) h w c")

        locations_mu, locations_sigma = self.regressor(images)
        locations_mu = E.rearrange(
            locations_mu, "(b t) o dim -> b t o dim", t=self.horizon
        )
        locations_sigma = E.rearrange(
            locations_sigma, "(b t) o dim -> b t o dim", t=self.horizon
        )

        locations = tfd.Normal(locations_mu, locations_sigma)
        locations = locations.sample(self.n_samples)
        locations = E.rearrange(locations, "n b t o dim -> b t o dim n")

        locations_t = locations[:, 0]
        reset_location = locations[:, 0]

        types_t = tf.zeros([locations_t.shape[0], self.n_objects, 3])
        types_t = tfd.Categorical(logits=types_t)
        types_t = types_t.sample(self.n_samples)
        types_t = E.rearrange(types_t, "n b o -> b o 1 n")

        held = inputs[1]
        observation = tf.ones(
            [locations_t.shape[0], self.n_objects, 2, self.n_samples], dtype=tf.int32
        )

        logits = 0
        for t in range(self.horizon - 1):
            locations_t = self.location_transition(
                locations_t,
                types_t,
                held[:, t],
                reset_location,
            )

            consistency_t = self.consistency(locations_t, locations[:, t + 1])
            consistency_t = self.get_bernoulli_parameters(consistency_t)

            unconditioned_samples = [locations_t, types_t]
            conditioned_samples, resampling_weights = self.infinite_resampler(
                unconditioned_samples, "categorical", consistency_t, observation
            )
            resampling_weights = E.reduce(resampling_weights, "b ... n -> b n", "sum")

            locations_t = conditioned_samples[0]
            types_t = conditioned_samples[1]
            logits += resampling_weights

        return types_t, locations[:, 0], logits
