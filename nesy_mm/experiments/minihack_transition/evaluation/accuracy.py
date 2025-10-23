import tensorflow as tf

POS_FREQ = 0.838


# @tf.function
def balanced_accuracy(model, batch, horizon, n_enemies, grid_size):
    actions = batch[0]
    agent_start_loc = batch[1]
    enemy_hit = batch[2]

    dead = batch[3]
    dead = dead[:, 0]
    dead = tf.cast(dead, tf.float32)

    dead_hat, _ = model([actions, agent_start_loc, enemy_hit, horizon, n_enemies, grid_size])
    dead_hat = tf.cast(dead_hat, tf.float32)
    dead_hat = tf.reduce_mean(dead_hat, axis=-1)
    dead_hat = tf.where(dead_hat < 0.5, 0.0, 1.0)

    sensitivity = tf.reduce_sum(dead * dead_hat)
    sensitivity = sensitivity / (tf.reduce_sum(dead) + tf.keras.backend.epsilon())

    specificity = tf.reduce_sum((1 - dead) * (1 - dead_hat))
    specificity = specificity / (tf.reduce_sum(1 - dead) + tf.keras.backend.epsilon())

    balanced_accuracy = (sensitivity + specificity) / 2
    return balanced_accuracy


def positive_accuracy(model, batch, horizon, n_enemies, grid_size):
    actions = batch[0]
    agent_start_loc = batch[1]
    enemy_hit = batch[2]

    dead = batch[3]
    dead = dead[:, 0]
    dead = tf.cast(dead, tf.float32)

    dead_hat, _ = model([actions, agent_start_loc, enemy_hit, horizon, n_enemies, grid_size])
    dead_hat = tf.cast(dead_hat, tf.float32)
    dead_hat = tf.reduce_mean(dead_hat, axis=-1)
    dead_hat = tf.where(dead_hat < 0.5, 0.0, 1.0)

    positive_accuracy = tf.reduce_sum(dead * dead_hat) / (
        tf.reduce_sum(dead) + tf.keras.backend.epsilon()
    )
    return positive_accuracy


def negative_accuracy(model, batch, horizon, n_enemies, grid_size):
    actions = batch[0]
    agent_start_loc = batch[1]
    enemy_hit = batch[2]

    dead = batch[3]
    dead = dead[:, 0]
    dead = tf.cast(dead, tf.float32)

    dead_hat, _ = model([actions, agent_start_loc, enemy_hit, horizon, n_enemies, grid_size])
    dead_hat = tf.cast(dead_hat, tf.float32)
    dead_hat = tf.reduce_mean(dead_hat, axis=-1)
    dead_hat = tf.where(dead_hat < 0.5, 0.0, 1.0)

    negative_accuracy = tf.reduce_sum((1 - dead) * (1 - dead_hat)) / (
        tf.reduce_sum(1 - dead) + tf.keras.backend.epsilon()
    )
    return negative_accuracy