import einops as E
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd

from nesy_mm.src.utils import add_batch_dimension_like


class InfiniteResampler(tf.Module):

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        unconditioned_samples,
        observation_distribution_name,
        observation_distribution_arguments,
        observation,
    ):
        """
        @param unconditioned_samples: list of samples (tensors) of the influenced variables
        @param observation_distribution_name: name of the distribution for the observation
        @param observation_distribution_arguments: parameters for the observation distribution
        @shape observation_distribution_arguments: (b, dim, samples, o_t)
        @param observations: the observation tensor

        @return: list of samples (tensors) of the influenced variables resampled to the observation
        """

        if observation_distribution_name != "categorical":
            raise NotImplementedError(
                f"Resampling for {observation_distribution_name} is not implemented."
            )
        else:
            batch_dims = len(observation_distribution_arguments.shape[:-1])
            weights = tf.gather(
                observation_distribution_arguments,
                observation,
                axis=-1,
                batch_dims=batch_dims,
            )
            weights = tf.reduce_sum(weights, axis=-2, keepdims=True)
            weights -= tf.reduce_logsumexp(weights, axis=-1, keepdims=True)

            n = weights.shape[-1]

            resampling_indices = tfd.Categorical(logits=weights)
            resampling_indices = resampling_indices.sample(n)
            resampling_indices = E.rearrange(resampling_indices, "n ... -> ... n")

            conditioned_samples = []
            for unconditioned_sample in unconditioned_samples:
                dim = unconditioned_sample.shape[-2]

                resampling_indices_local = E.repeat(
                    resampling_indices, "... 1 n -> ... dim n", dim=dim
                )

                conditioned_sample = tf.gather(
                    unconditioned_sample,
                    resampling_indices_local,
                    axis=-1,
                    batch_dims=batch_dims - 1,
                )
                conditioned_samples.append(conditioned_sample)
            return conditioned_samples, weights


class FiniteResampler(tf.Module):

    def __init__(self):
        super().__init__()

    def get_einops_pattern(self, combination_weights):
        combination_shape = list(combination_weights.shape[2:])
        combination_shape_names = " ".join(
            [f"c_{i}" for i in range(len(combination_shape))]
        )
        combination_mapping = dict(
            zip(
                combination_shape_names.split(),
                combination_shape,
            )
        )
        einops_pattern = (
            f"b ({combination_shape_names}) -> b 1 {combination_shape_names}"
        )
        return einops_pattern, combination_mapping

    def __call__(
        self,
        observation_distribution_name,
        observation_distribution_parameters,
        observation,
        combination_weights,
        combination_tensors,
    ):
        """
        Implementation of a Rao-Blackwellised particle filter for finite variables.

        @param observation_distribution_name: name of the distribution for the observation
        @param observation_distribution_parameters: parameters for the observation distribution for all combinations
        @shape observation_distribution_parameters: (b, dim_o, combinations, o_t)
        @param observations: the observation tensor
        @param combination_weights: list of logarithmic weights of each combination
        @shape combination_weights: (b, n, combinations_1 ... combinations_n)
        @param combination_tensors: list of tensors containing all possible combinations of the domains of the variables
        @shape combination_tensors: list of (dimension, combinations)

        @return: list of samples (tensors) of the influenced variables sampled directly from the exact conditional probabilities
        """

        if observation_distribution_name != "categorical":
            raise NotImplementedError(
                f"Resampling for {observation_distribution_name} is not implemented."
            )
        else:
            einops_pattern, combination_mapping = self.get_einops_pattern(
                combination_weights
            )

            observation_weights = tf.gather(
                observation_distribution_parameters,
                observation,
                axis=-1,
                batch_dims=3,
            )  # (b, dim_o, combinations)
            observation_weights = tf.reduce_sum(
                observation_weights, axis=-2
            )  # (b, combinations)
            observation_weights = E.rearrange(
                observation_weights,
                einops_pattern,
                **combination_mapping,
            )  # (b, 1, combinations_1 ... combinations_n)

            exact_conditional = (
                observation_weights + combination_weights
            )  # (b, n, combinations_1 ... combinations_n)

            combination_weights = E.rearrange(combination_weights, "b ... -> b (...)")

            n = len(exact_conditional.shape) - 2

            exact_conditional_samples = []
            exact_conditionals = []
            for i in range(n):
                combination_i = add_batch_dimension_like(
                    combination_tensors[i], combination_weights[:, 0]
                )  # (b, dim_i, combinations_i)

                reduction_axes = list(range(2, 2 + i))
                reduction_axes += list(range(3 + i, 3 + n - 1))

                exact_conditional_i = tf.reduce_logsumexp(
                    exact_conditional, axis=reduction_axes
                )  # (b, n, combinations_i)

                exact_conditional_sample = tfd.Categorical(logits=exact_conditional_i)
                exact_conditional_sample = exact_conditional_sample.sample(1)[0]

                exact_conditional_i = tf.gather(
                    exact_conditional_i,
                    exact_conditional_sample,
                    batch_dims=2,
                )

                exact_conditional_sample = E.repeat(
                    exact_conditional_sample,
                    "b n -> b dim n",
                    dim=combination_tensors[i].shape[-2],
                )
                exact_conditional_sample = tf.gather(
                    combination_i,
                    exact_conditional_sample,
                    batch_dims=2,
                )

                exact_conditional_samples.append(exact_conditional_sample)
                exact_conditionals.append(exact_conditional_i)

            return exact_conditional_samples, exact_conditionals
