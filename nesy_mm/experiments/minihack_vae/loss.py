import einops as E
import tensorflow as tf


EPS = tf.keras.backend.epsilon()


class LocationLogProbability(tf.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, targets, inputs, training=False, mask=None):
        targets = E.rearrange(targets, "b d -> b d 1")

        indicator = tf.where(targets == inputs, 1.0, 0.0)
        indicator = tf.reduce_prod(indicator, axis=-2)
        indicator = tf.where(indicator < EPS, indicator + EPS, indicator)
        indicator = tf.cast(indicator, tf.keras.backend.floatx())
        return indicator


class ImageCrossEntropy(tf.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, targets, images, training=False, mask=None):
        targets = E.rearrange(targets, "b t w h c -> b t 1 (w h c)")
        images = E.rearrange(images, "b t n w h c -> b t n (w h c)")
        images = tf.where(images < EPS, images + EPS, images)

        negative_images = 1 - images
        negative_images = tf.where(
            negative_images < EPS, negative_images + EPS, negative_images
        )

        cross_entropy = -targets * tf.math.log(images) - (1 - targets) * tf.math.log(
            negative_images
        )
        cross_entropy = tf.reduce_sum(cross_entropy, axis=-1)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=1)
        return cross_entropy  # (b, n)


class GaussianKLDLoss(tf.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, mu, sigma, training=False, mask=None):
        gaussian_kld = sigma + tf.square(mu) - 1 - tf.math.log(sigma)
        gaussian_kld = tf.reduce_sum(gaussian_kld, axis=-1, keepdims=True)
        gaussian_kld = 0.5 * gaussian_kld
        return gaussian_kld


class MinihackLoss(tf.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def __call__(self, targets, inputs, training=False, mask=None):
        generations = inputs[0]
        mu, sigma = inputs[1]
        agent_t = inputs[2]

        images = targets[0]
        location = targets[1]

        classification_loss = LocationLogProbability()
        classification_loss = classification_loss(location, agent_t)
        classification_loss = classification_loss

        reconstruction_loss = ImageCrossEntropy()
        reconstruction_loss = reconstruction_loss(images, generations)
        reconstruction_loss = tf.exp(-self.gamma * reconstruction_loss)

        kld_loss = GaussianKLDLoss()
        kld_loss = kld_loss(mu, sigma)
        kld_loss = tf.exp(-self.gamma * self.beta * kld_loss)

        loss = classification_loss * reconstruction_loss * kld_loss
        return loss


class MinihackClassificationLoss(tf.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, targets, inputs, training=False, mask=None):
        agent_t = inputs[2]
        location = targets[1]

        classification_loss = LocationLogProbability()
        classification_loss = classification_loss(location, agent_t)
        return classification_loss
