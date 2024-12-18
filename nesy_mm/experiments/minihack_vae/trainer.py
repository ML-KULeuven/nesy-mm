import tensorflow as tf
import matplotlib.pyplot as plt
import einops as E
import wandb
from PIL import Image
import io

from nesy_mm.src.trainer import Trainer


class MinihackVAETrainer(Trainer):
    def __init__(
        self,
        model,
        optimiser,
        loss,
        estimator,
        evaluation_functions,
        log_its=100,
        test_data=None,
        checkpoint_epochs=5,
        wandb_on=True,
        save_dir="",
        model_name="",
    ):
        super(MinihackVAETrainer, self).__init__(
            model,
            optimiser,
            loss,
            log_its=log_its,
            test_data=test_data,
            checkpoint_epochs=checkpoint_epochs,
            wandb_on=wandb_on,
            save_dir=save_dir,
            model_name=model_name,
        )
        self.estimator = estimator
        self.evaluation_functions = evaluation_functions

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            images = batch[0]
            actions = batch[1]
            location = batch[2]

            output = self.model([images, actions])

            loss = self.loss([images, location], output[:-1])
            loss, grads = self.estimator([loss, output[-1]])
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

    def checkpoint(self, epoch):
        super(MinihackVAETrainer, self).checkpoint(epoch)

        if self.wandb_on:
            # Generate 5 image samples for sequence of actions "right, down, left, up, left, up, right, down"
            # 0: up, 1: right, 2: down, 3: left
            actions_list = [1, 2, 3, 0, 3, 0, 1, 2]
            actions = tf.constant(actions_list, dtype=tf.int32)

            # loc = tf.constant([3, 3], dtype=tf.int32)
            generations = self.model.generate(actions, 1, starting_loc=None)
            generations = E.rearrange(generations, "1 t s h w c -> s t h w c")
            generation = generations[0]

            # Desired size in pixels
            height_px = 400
            width_px = height_px * len(actions_list)

            # DPI (dots per inch)
            dpi = 100

            # Convert pixels to inches
            width_in = width_px / dpi
            height_in = height_px / dpi

            image = [img for img in generation]
            image = tf.concat(image, axis=1)

            plt.figure(figsize=(width_in, height_in), dpi=dpi)
            plt.imshow(image)
            plt.axis("off")
            # Save the plot to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            # Read the buffer into a PIL image
            image = Image.open(buf)
            wandb.log({f"generation_sample": wandb.Image(image)})
            # Close the buffer
            buf.close()
