import tensorflow as tf


# @tf.function
def f1_score(model, batch, horizon, n_enemies, grid_size):
    actions = batch[0]
    agent_start_loc = batch[1]
    enemy_hit = batch[2]

    dead = batch[3]
    dead = dead[:, 0]
    dead = tf.cast(dead, tf.float32)

    dead_hat, _ = model(
        [actions, agent_start_loc, enemy_hit, horizon, n_enemies, grid_size]
    )
    dead_hat = tf.cast(dead_hat, tf.float32)
    dead_hat = tf.reduce_mean(dead_hat, axis=-1)
    dead_hat = tf.where(dead_hat < 0.5, 0.0, 1.0)

    true_positives = tf.reduce_sum(dead * dead_hat)
    false_positives = tf.reduce_sum((1 - dead) * dead_hat)
    false_negatives = tf.reduce_sum(dead * (1 - dead_hat))

    precision = true_positives / (
        true_positives + false_positives + tf.keras.backend.epsilon()
    )
    recall = true_positives / (
        true_positives + false_negatives + tf.keras.backend.epsilon()
    )

    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1