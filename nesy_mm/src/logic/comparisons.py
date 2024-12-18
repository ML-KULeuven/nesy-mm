import tensorflow as tf


EPS = tf.keras.backend.epsilon()


class SmallerThan(tf.Module):
    def __init__(self, log_space):
        super().__init__()
        self.factor = 2.0
        # self.factor = 10.0
        self.log_space = log_space

    def __call__(self, x, y, training=False, mask=None):
        difference = self.factor * (y - x)
        if self.log_space:
            return tf.math.log_sigmoid(difference)
        else:
            return tf.math.sigmoid(difference)


class EqualTo(tf.Module):
    def __init__(self, log_space):
        super().__init__()
        self.log_space = log_space
        self.factor = 1.0
        # self.factor = 5.0

    def __call__(self, x, y, training=False, mask=None):
        difference = self.factor * (y - x)
        if self.log_space:
            equals = (
                tf.math.log(tf.constant(4.0, dtype=tf.keras.backend.floatx()))
                + tf.math.log_sigmoid(2 * difference)
                + tf.math.log_sigmoid(-2 * difference)
            )
        else:
            equals = (
                4 * tf.math.sigmoid(2 * difference) * tf.math.sigmoid(-2 * difference)
            )

        # equals = tf.square(x - y)
        # equals = tf.exp(-equals)
        return equals
