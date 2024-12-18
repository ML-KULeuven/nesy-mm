import tensorflow as tf

from nesy_mm.src.trainer import Trainer


class TransformerTrainer(Trainer):
    def __init__(
        self,
        model,
        optimizer,
        loss,
        evaluation_functions,
        test_data,
        save_dir,
        model_name,
        checkpoint_epochs,
        wandb_on=True,
        log_its=100,
    ):
        super().__init__(
            model,
            optimizer,
            loss,
            test_data=test_data,
            save_dir=save_dir,
            model_name=model_name,
            checkpoint_epochs=checkpoint_epochs,
            wandb_on=wandb_on,
            log_its=log_its,
        )
        self.evaluation_functions = evaluation_functions

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            actions = batch[0]
            agent_start_loc = batch[1]
            enemy_hit = batch[2]
            dead = batch[3]

            dead_hat = self.model([actions, agent_start_loc, enemy_hit])
            loss = self.loss(dead, dead_hat)
            grads = tape.gradient(loss, self.model.trainable_variables)
        return loss, grads

    def evaluation(self, evaluation_data):
        evaluations = dict()
        for name, evaluation_function in self.evaluation_functions.items():
            evaluations[name] = tf.keras.metrics.Mean()
            for batch in evaluation_data:
                evaluation = evaluation_function(self.model, batch)
                evaluations[name].update_state(evaluation)

        evaluations["validation_loss"] = tf.keras.metrics.Mean()
        for batch in evaluation_data:
            actions = batch[0]
            agent_start_loc = batch[1]
            enemy_hit = batch[2]
            dead = batch[3]

            dead_hat = self.model([actions, agent_start_loc, enemy_hit])
            loss = self.loss(dead, dead_hat)
            evaluations["validation_loss"].update_state(loss)
        return evaluations
