import time
import tensorflow as tf
import wandb
import os


class Trainer:
    def __init__(
        self,
        model,
        optimiser,
        loss,
        log_its=100,
        test_data=None,
        checkpoint_epochs=None,
        wandb_on=True,
        save_dir="",
        model_name="",
    ):
        self.model = model
        self.optimiser = optimiser
        self.loss = loss
        self.log_its = log_its
        self.test_data = test_data
        self.checkpoint_epochs = checkpoint_epochs
        self.wandb_on = wandb_on
        self.save_dir = save_dir
        self.model_name = model_name

    def train_step(self, batch) -> tuple:
        pass

    def evaluation(self, evaluation_data) -> dict:
        pass

    def test(self, test_data) -> dict:
        evaluations = self.evaluation(test_data)
        tests = {}
        for name, evaluation in evaluations.items():
            tests[f"test_{name}"] = evaluation.result().numpy()

        if self.wandb_on:
            wandb.log(tests)

        return tests

    def checkpoint(self, epoch):
        self.test(self.test_data)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        model_name = f"{self.model_name}_checkpoint{epoch}.h5"
        self.model.save_weights(os.path.join(self.save_dir, model_name))
        if self.wandb_on:
            self.model.save_weights(os.path.join(wandb.run.dir, model_name))

    def train(self, train_data, n_epochs, validation_data=None):
        training_time = tf.keras.metrics.Sum()
        training_loss = tf.keras.metrics.Mean()

        iteration = 0
        for epoch in range(1, 1 + n_epochs):
            for batch in train_data:
                start_time = time.time()
                loss, grads = self.train_step(batch)
                self.optimiser.apply_gradients(
                    zip(grads, self.model.trainable_variables)
                )

                training_time.update_state(time.time() - start_time)
                training_loss.update_state(loss)

                if iteration % self.log_its == 0:
                    printing_text = []
                    printing_text += ["Epoch", epoch]
                    printing_text += ["\tIteration", iteration]
                    printing_text += ["\tTime:", training_time.result().numpy(), "(s)"]
                    printing_text += [
                        "\tTraining Loss:",
                        training_loss.result().numpy(),
                    ]

                    logging_dict = {
                        "train_loss": training_loss.result().numpy(),
                        "train_time": training_time.result().numpy(),
                    }

                    if validation_data is not None:
                        evaluations = self.evaluation(validation_data)
                        for name, evaluation in evaluations.items():
                            printing_text += [
                                f"\t{name}:",
                                evaluation.result().numpy(),
                            ]

                            logging_dict[name] = evaluation.result().numpy()
                    print(*printing_text)

                    if self.wandb_on:
                        # Add parameter gradients to logging dict
                        # for param, grad in zip(self.model.trainable_variables, grads):
                        #     key = param.name
                        #     key = key.replace(":", "-")
                        #     key = key.replace("/", "-")
                        #     key = f"Gradients/{key}"
                        #     logging_dict[key] = tf.reduce_mean(grad).numpy()

                        wandb.log(logging_dict)

                    training_time.reset_states()
                    training_loss.reset_states()
                iteration += 1

            if self.checkpoint_epochs is not None:
                if epoch % self.checkpoint_epochs == 0:
                    self.checkpoint(epoch)
