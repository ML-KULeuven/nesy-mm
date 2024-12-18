import tensorflow as tf


EPS = tf.keras.backend.epsilon()


class Estimator(tf.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs, training=False, mask=None):
        pass


class RLOO(Estimator):
    def __init__(self, elbo=False, discrete_grad_weight=1.0):
        super().__init__()
        self.elbo = elbo
        self.discrete_grad_weight = tf.cast(
            discrete_grad_weight, tf.keras.backend.floatx()
        )

    def __call__(self, inputs, training=False, mask=None):
        loss = inputs[0]
        logits = inputs[1]

        n = logits.shape[-1]
        mean_loss = tf.reduce_mean(loss, axis=-1, keepdims=True)
        mean_loss = tf.clip_by_value(mean_loss, EPS, 1 - EPS)

        rloo = tf.stop_gradient(loss - mean_loss) * logits
        rloo = tf.reduce_sum(rloo, axis=-1) / (n - 1)

        if not self.elbo:
            discrete_grads = -rloo / tf.stop_gradient(mean_loss[:, 0])
        else:
            discrete_grads = rloo
        discrete_grads = tf.reduce_mean(discrete_grads)

        continuous_grads = mean_loss[:, 0]
        if not self.elbo:
            continuous_grads = tf.where(
                continuous_grads < EPS, continuous_grads + EPS, continuous_grads
            )
            continuous_grads = -tf.math.log(continuous_grads)
        continuous_grads = tf.reduce_mean(continuous_grads)

        grads = ((self.discrete_grad_weight * discrete_grads) + continuous_grads) / 2
        return continuous_grads, grads
