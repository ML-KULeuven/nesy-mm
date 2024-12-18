import tensorflow as tf

from nesy_mm.src.trainer import Trainer


class MinihackTransitionTrainer(Trainer):
    def __init__(
        self,
        model,
        optimizer,
        loss,
        estimator,
        evaluation_functions,
        horizon,
        test_horizon,
        n_enemies,
        n_test_enemies,
        grid_size,
        test_grid_size,
        test_data,
        save_dir,
        model_name,
        checkpoint_epochs,
        wandb_on=True,
        log_its=100,
    ):
        super(MinihackTransitionTrainer, self).__init__(
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
        self.estimator = estimator
        self.evaluation_functions = evaluation_functions
        self.horizon = horizon
        self.test_horizon = test_horizon
        self.n_enemies = n_enemies
        self.n_test_enemies = n_test_enemies
        self.grid_size = grid_size
        self.test_grid_size = test_grid_size

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            actions = batch[0]
            agent_start_loc = batch[1]
            enemy_hit = batch[2]
            dead = batch[3]

            dead_hat, logits = self.model(
                [
                    actions,
                    agent_start_loc,
                    enemy_hit,
                    self.horizon,
                    self.n_enemies,
                    self.grid_size,
                ]
            )
            loss = self.loss(dead, dead_hat)
            loss, grads = self.estimator([loss, logits])
            grads = tape.gradient(grads, self.model.trainable_variables)
        return loss, grads

    def evaluation(self, evaluation_data):
        evaluations = dict()
        for name, evaluation_function in self.evaluation_functions.items():
            evaluations[name] = tf.keras.metrics.Mean()
            for batch in evaluation_data:
                evaluation = evaluation_function(
                    self.model,
                    batch,
                    self.test_horizon,
                    self.n_test_enemies,
                    self.test_grid_size,
                )
                evaluations[name].update_state(evaluation)

        evaluations["validation_loss"] = tf.keras.metrics.Mean()
        for batch in evaluation_data:
            actions = batch[0]
            agent_start_loc = batch[1]
            enemy_hit = batch[2]
            dead = batch[3]

            dead_hat, _ = self.model(
                [
                    batch[0],
                    batch[1],
                    batch[2],
                    self.test_horizon,
                    self.n_test_enemies,
                    self.test_grid_size,
                ]
            )
            loss = self.loss(dead, dead_hat)
            loss = tf.reduce_mean(loss, -1)
            loss = tf.reduce_mean(loss)
            evaluations["validation_loss"].update_state(loss)
        return evaluations
