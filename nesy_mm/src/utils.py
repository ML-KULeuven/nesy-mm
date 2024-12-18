import einops as E
import tensorflow as tf


EPS = tf.keras.backend.epsilon()


def log1mexp(x):
    """
    Computing the true log1mexp function: log(1 - exp(x))
    """

    logand = -tf.math.expm1(x)
    logand = tf.clip_by_value(logand, EPS, 1 - EPS)
    log1mexp = tf.math.log(logand)
    return log1mexp


def add_batch_dimension_like(tensor, batch_tensor):
    """
    Add a batch dimension to a tensor
    """

    multiplier_example = batch_tensor
    multiplier = tf.ones_like(multiplier_example, dtype=tensor.dtype)
    multiplier = tf.expand_dims(multiplier, axis=0)

    tensor = E.rearrange(tensor, "... -> ... 1")
    tensor = tf.matmul(tensor, multiplier)
    tensor = E.rearrange(tensor, "... b -> b ...")
    return tensor


def get_bernoulli_parameters(parameters, log_space):
    if log_space:
        logits_1 = parameters
        logits_0 = log1mexp(parameters)
    else:
        logits_1 = tf.clip_by_value(parameters, EPS, 1 - EPS)
        logits_0 = 1 - logits_1
        logits_0 = tf.math.log(logits_0)
        logits_1 = tf.math.log(logits_1)

    logits = tf.stack([logits_0, logits_1], axis=-1)
    return logits
