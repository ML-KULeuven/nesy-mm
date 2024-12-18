import einops as E
import tensorflow as tf

from nesy_mm.src.utils import add_batch_dimension_like, get_bernoulli_parameters

EPS = tf.keras.backend.epsilon()


class CombinationConstructor(tf.keras.layers.Layer):

    def __init__(self, name=None):
        super(CombinationConstructor, self).__init__()

    def call(self, distribution_parameters, domains):
        """
        Preparation for tensorised exact probabilistic inference.

        @param distribution_parameters: list of parameters of variables influencing the end result (observation)
        @shape distribution_parameters: (b, dimension, n, domain)
        @param domains: list of domains of the variables

        @return:
        1. list of tensors containing all possible combinations of the domains of the variables (exponential!)
        2. tensor containing the weights of the combinations in format (b n combinations_1 ... combinations_n)
        """

        n = len(distribution_parameters)

        dimensions = tf.constant(
            [distribution_parameters[i].shape[-3] for i in range(n)], dtype=tf.int64
        )

        combination_tensors = []
        combination_weights = tf.zeros(
            [
                1,
                1,
            ]
            + [1] * n,
            dtype=tf.keras.backend.floatx(),
        )

        domain_sizes = []
        for i in range(n):
            if type(domains[i]) != list:
                domain = [0, 1]
                domain = tf.constant(domain, dtype=tf.int64)
            else:
                domain = domains[i]
                domain = tf.stack(domain, axis=0)
            domains[i] = domain
            domain_sizes.append(len(domain))
        domain_sizes = tf.constant(domain_sizes, dtype=tf.int64)

        for i in range(n):
            domain = domains[i]
            domain = E.repeat(domain, "d -> dimension d", dimension=dimensions[i])
            domain = tf.unstack(domain, axis=0)

            parameter_indices = tf.range(len(domain[0]), dtype=tf.int64)
            parameter_indices = E.repeat(
                parameter_indices, "d -> dimension d", dimension=dimensions[i]
            )
            parameter_indices = tf.unstack(parameter_indices, axis=0)

            parameters = distribution_parameters[i]
            # parameters = tf.math.log(parameters + EPS)
            parameters = E.rearrange(
                parameters, "b dimension n domain -> b n dimension domain"
            )
            if parameters.shape[-1] == 1:
                parameters = get_bernoulli_parameters(parameters, log_space=True)

            combination_indices = tf.meshgrid(*parameter_indices, indexing="ij")
            combination_indices = tf.stack(combination_indices, axis=-1)
            combination_indices = E.repeat(
                combination_indices,
                "... dimension -> n dimension (...)",
                n=parameters.shape[1],
            )
            combination_indices = add_batch_dimension_like(
                combination_indices, parameters[:, 0, 0, 0]
            )

            combination_parameters = tf.gather(
                parameters,
                combination_indices,
                batch_dims=3,
            )
            combination_parameters = tf.reduce_sum(combination_parameters, axis=-2)

            final_form_i = [-1]
            final_form_i += list(combination_parameters.shape[1:2])
            final_form_i += [1] * (i)
            final_form_i += [combination_parameters.shape[-1]]
            final_form_i += [1] * (n - i - 1)

            combination_parameters = tf.reshape(combination_parameters, final_form_i)

            dimension_until_i = domain_sizes[:i] ** dimensions[:i]
            dimension_until_i = tf.reduce_prod(dimension_until_i)

            dimension_past_i = domain_sizes[i + 1 :] ** dimensions[i + 1 :]
            dimension_past_i = tf.reduce_prod(dimension_past_i)

            combination_tensor = tf.meshgrid(*domain, indexing="ij")
            combination_tensor = tf.stack(combination_tensor, axis=-1)
            combination_tensor = E.rearrange(
                combination_tensor, "... dimension -> (...) dimension"
            )
            combination_tensor = E.repeat(
                combination_tensor,
                "combinations_i dimension -> before_i combinations_i past_i dimension",
                before_i=dimension_until_i,
                past_i=dimension_past_i,
            )
            combination_tensor = E.rearrange(
                combination_tensor, "... dimension -> dimension (...)"
            )

            combination_tensors.append(combination_tensor)
            combination_weights = combination_weights + combination_parameters

        return [combination_tensors, combination_weights]
