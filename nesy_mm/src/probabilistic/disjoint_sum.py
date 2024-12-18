import einops as E
import tensorflow as tf


def disjoint_sum(events, axis=-1):
    if axis != -1:
        raise NotImplementedError("Only axis=-1 is supported for now.")

    n_axes = len(events.shape) - 1
    n_events = events.shape[-1]

    combinations = tf.range(2)
    combinations = E.repeat(combinations, "b -> n b", n=n_events)
    combinations = tf.unstack(combinations, axis=0)
    combinations = tf.meshgrid(*combinations, indexing="ij")
    combinations = tf.stack(combinations, axis=0)
    combinations = E.rearrange(combinations, "n ... -> n (...)")
    for _ in range(n_axes):
        combinations = tf.expand_dims(combinations, axis=0)

    true = tf.reduce_sum(combinations, axis=-2)
    true = true > 0
    true = tf.cast(true, tf.float32)

    disjoint_sum = tf.expand_dims(events, axis=-1)
    disjoint_sum = tf.where(combinations == 1, disjoint_sum, 1 - disjoint_sum)
    disjoint_sum = tf.reduce_prod(disjoint_sum, axis=-2)
    disjoint_sum = disjoint_sum * true
    disjoint_sum = tf.reduce_sum(disjoint_sum, axis=-1)
    return disjoint_sum
