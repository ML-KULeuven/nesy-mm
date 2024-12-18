import tensorflow as tf

EPS = tf.keras.backend.epsilon()


def variational_transformer_loss(beta):
    categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False
    )

    def loss(target_images, generations, priors, final_mario):
        entropy = image_binary_crossentropy(target_images, generations)
        kl = kl_divergence(priors)
        kl = kl[:, 0]
        label_loss = categorical_crossentropy(final_mario, final_mario)

        loss = entropy - beta * kl
        loss = loss / (125 * 125 * 3)
        # loss = tf.reduce_sum(loss, -1)
        loss -= label_loss
        loss = tf.reduce_mean(loss)

        return -loss

    return loss


def image_binary_crossentropy(y_true, y_pred):
    y_pred = tf.where(y_pred < EPS, y_pred + EPS, y_pred)

    negative_y_pred = 1 - y_pred
    negative_y_pred = tf.where(
        negative_y_pred < EPS, negative_y_pred + EPS, negative_y_pred
    )

    cross_entropy = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(
        negative_y_pred
    )
    cross_entropy = tf.reduce_sum(cross_entropy, axis=[-3, -2, -1])
    cross_entropy = tf.reduce_sum(cross_entropy, axis=-1)

    return cross_entropy


def kl_divergence(priors):
    mu = priors[..., 0]
    sigma = priors[..., 1]

    gaussian_kl = sigma + tf.square(mu) - 1 - tf.math.log(sigma + EPS)
    gaussian_kl = tf.reduce_sum(gaussian_kl, axis=-1)
    gaussian_kl = 0.5 * gaussian_kl

    return gaussian_kl
