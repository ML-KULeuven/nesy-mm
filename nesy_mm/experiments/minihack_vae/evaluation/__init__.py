import einops as E
import tensorflow as tf
import os

from nesy_mm.experiments.minihack_vae.neural.classifier import AgentClassifier


# @tf.function
def classifier_accuracy(model, batch):
    images = batch[0]
    images = E.rearrange(images, "b t w h c -> (b t) w h c")
    locations = batch[2]
    locations = locations[:, 0]

    labels = model.classifier(images)
    labels = E.rearrange(labels, "(b t) d l -> b t d l", t=model.horizon)
    labels = labels[:, 0]
    labels = tf.argmax(labels, axis=-1)
    labels = tf.cast(labels, tf.int32)

    accuracy = tf.equal(labels, locations)
    accuracy = tf.reduce_all(accuracy, axis=-1)
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy, axis=-1)
    accuracy = tf.reduce_mean(accuracy)
    return accuracy


# @tf.function
def accuracy_call(model, batch, classifier):
    images = batch[0]
    actions = batch[1]
    locations = batch[2]

    output = model([images, actions])
    generations = output[0]
    generations = generations[:, :, 0]
    generations = E.rearrange(generations, "b t w h c -> (b t) w h c")

    labels = classifier(generations)
    labels = E.rearrange(labels, "(b t) d l -> b t d l", t=model.horizon)
    labels = tf.argmax(labels, axis=-1)
    labels = tf.cast(labels, tf.int32)

    accuracy = tf.equal(labels, locations)
    accuracy = tf.reduce_all(accuracy, axis=-1)
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy, axis=-1)
    accuracy = tf.reduce_mean(accuracy)
    return accuracy


def generative_accuracy(model, batch):
    classifier = AgentClassifier(model.grid_size, model.image_shape)
    classifier.build((None,) + model.image_shape)
    downsample = model.downsample
    downsample_str = f"_downsample{downsample}" if downsample > 1 else ""
    model_weights_name = f"location_classifier{model.grid_size}x{model.grid_size}{downsample_str}_weights.h5"
    classifier.load_weights(
        os.path.join("nesy_mm/experiments/minihack_vae/evaluation/", model_weights_name)
    )

    accuracy = accuracy_call(model, batch, classifier)

    del classifier

    return accuracy


# @tf.function
def reconstruction_error(model, batch):
    images = batch[0]
    actions = batch[1]

    output = model([images, actions])
    generations = output[0]
    generations = generations[:, :, 0]
    generations = E.rearrange(generations, "b t w h c -> (b t) w h c")

    images = E.rearrange(images, "b t w h c -> (b t) w h c")
    error = tf.abs(images - generations)
    error = tf.reduce_mean(error, axis=[-1, -2, -3])
    error = tf.reduce_mean(error, -1)
    error = tf.reduce_mean(error)
    return error


# @tf.function
def encoder_predictions(model, batch):
    images = batch[0]
    actions = batch[1]

    output = model([images, actions])
    mu, sigma = output[1]

    mu = tf.reduce_mean(mu, axis=-1)
    mu = tf.reduce_mean(mu, axis=-1)

    sigma = tf.reduce_mean(sigma, axis=-1)
    sigma = tf.reduce_mean(sigma, axis=-1)

    return sigma
