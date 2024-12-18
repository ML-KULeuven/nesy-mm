import einops as E
import tensorflow as tf


def balanced_accuracy(model, batch):
    actions = batch[0]
    agent_start_loc = batch[1]
    enemy_hit = batch[2]

    dead = batch[3]
    dead = dead[:, 0]
    dead = tf.cast(dead, tf.float32)

    dead_hat = model([actions, agent_start_loc, enemy_hit])
    dead_hat = dead_hat[:, 0]
    dead_hat = tf.where(dead_hat < 0.5, 0.0, 1.0)

    sensitivity = tf.reduce_sum(dead * dead_hat)
    sensitivity = sensitivity / (tf.reduce_sum(dead) + tf.keras.backend.epsilon())

    specificity = tf.reduce_sum((1 - dead) * (1 - dead_hat))
    specificity = specificity / (tf.reduce_sum(1 - dead) + tf.keras.backend.epsilon())

    balanced_accuracy = (sensitivity + specificity) / 2
    return balanced_accuracy


def positive_accuracy(model, batch):
    actions = batch[0]
    agent_start_loc = batch[1]
    enemy_hit = batch[2]

    dead = batch[3]
    dead = dead[:, 0]
    dead = tf.cast(dead, tf.float32)

    dead_hat = model([actions, agent_start_loc, enemy_hit])
    dead_hat = dead_hat[:, 0]
    dead_hat = tf.where(dead_hat < 0.5, 0.0, 1.0)

    positive_accuracy = tf.reduce_sum(dead * dead_hat)
    positive_accuracy = positive_accuracy / (
        tf.reduce_sum(dead) + tf.keras.backend.epsilon()
    )
    return positive_accuracy


def negative_accuracy(model, batch):
    actions = batch[0]
    agent_start_loc = batch[1]
    enemy_hit = batch[2]

    dead = batch[3]
    dead = dead[:, 0]
    dead = tf.cast(dead, tf.float32)

    dead_hat = model([actions, agent_start_loc, enemy_hit])
    dead_hat = dead_hat[:, 0]
    dead_hat = tf.where(dead_hat < 0.5, 0.0, 1.0)

    negative_accuracy = tf.reduce_sum((1 - dead) * (1 - dead_hat))
    negative_accuracy = negative_accuracy / (
        tf.reduce_sum(1 - dead) + tf.keras.backend.epsilon()
    )
    return negative_accuracy


def f1_score(model, batch):
    actions = batch[0]
    agent_start_loc = batch[1]
    enemy_hit = batch[2]

    dead = batch[3]
    dead = dead[:, 0]
    dead = tf.cast(dead, tf.float32)

    dead_hat = model([actions, agent_start_loc, enemy_hit])
    dead_hat = dead_hat[:, 0]
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
