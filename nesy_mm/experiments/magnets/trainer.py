import tensorflow as tf

from nesy_mm.src.trainer import Trainer


class MagnetTrainer(Trainer):
    def __init__(
        self, model, optimizer, loss, estimator, evaluation_functions, log_its=100
    ):
        super(MagnetTrainer, self).__init__(model, optimizer, loss, log_its)
        self.estimator = estimator
        self.evaluation_functions = evaluation_functions

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            images = batch[0]
            held = batch[1]
            types = batch[2]
            locations = batch[3]

            types_hat, locations_hat, logits = self.model([images, held])

            loss = self.loss([types, locations], [types_hat, locations_hat])
            loss, grads = self.estimator([loss, logits])
            grads = tape.gradient(grads, self.model.trainable_variables)
            # grads = [
            #     grad / (tf.norm(grad) + 1e-8) if grad is not None else None
            #     for grad in grads
            # ]
        return loss, grads

    def evaluation(self, evaluation_data):
        evaluations = dict()
        for name, evaluation_function in self.evaluation_functions.items():
            evaluations[name] = tf.keras.metrics.Mean()
            for batch in evaluation_data:
                evaluation = evaluation_function(self.model, batch)
                evaluations[name].update_state(evaluation)
        return evaluations
