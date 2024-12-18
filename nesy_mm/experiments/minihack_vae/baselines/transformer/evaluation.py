import tensorflow as tf
import einops as E
import os

from nesy_mm.experiments.minihack_vae.neural.classifier import AgentClassifier


def reconstruction_error(model, batch):
    images = batch[0]
    actions = batch[1]
    actions = tf.one_hot(actions, depth=4)

    predictions = model(images, actions)
    generations = predictions[0]

    error = tf.abs(images - generations)
    error = tf.reduce_mean(error, axis=[-1, -2, -3])
    error = tf.reduce_mean(error, -1)
    error = tf.reduce_mean(error)

    return error


def generative_accuracy(model, batch):
    classifier = AgentClassifier(model.grid_size, model.image_shape)
    classifier.build((None,) + model.image_shape)
    downsample = model.downsample
    downsample_str = f"_downsample{downsample}" if downsample > 1 else ""
    model_weights_name = f"location_classifier{model.grid_size}x{model.grid_size}{downsample_str}_weights.h5"
    classifier.load_weights(
        os.path.join("nesy_mm/experiments/minihack_vae/evaluation/", model_weights_name)
    )

    images = batch[0]
    actions = batch[1]
    actions = tf.one_hot(actions, depth=4)
    locations = batch[2]

    generations = model(images, actions)
    generations = generations[0]
    generations = E.rearrange(generations, "b t w h c -> (b t) w h c")

    labels = classifier(generations)
    labels = E.rearrange(labels, "(b t) d l -> b t d l", t=model.horizon)
    labels = tf.argmax(labels, axis=-1)
    labels = tf.cast(labels, tf.int32)

    accuracy = tf.equal(labels, locations)
    accuracy = tf.reduce_all(accuracy, axis=-1)
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy)

    del classifier

    return accuracy
