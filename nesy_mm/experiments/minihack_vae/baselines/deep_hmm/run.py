import os
import wandb
import keras.metrics
import tensorflow as tf

from argparse import ArgumentParser
from nesy_mm.src.estimators import RLOO
from nesy_mm.experiments.minihack_vae.data.generator import create_loader
from nesy_mm.experiments.minihack_vae.baselines.deep_hmm.model import (
    MinihackVAEModelHMM,
)
from nesy_mm.experiments.minihack_vae.loss import MinihackLoss
from nesy_mm.experiments.minihack_vae.trainer import MinihackVAETrainer
from nesy_mm.experiments.minihack_vae.evaluation import *

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def run(
    grid_size,
    n_samples,
    latent_dim,
    dropout,
    downsample,
    seed,
    batch_size,
    horizon,
    n_epochs,
    alpha,
    beta,
    learning_rate,
    discrete_grad_weight,
    checkpoint,
    base_dir="",
    wandb_mode="disabled",
    weights_path=None,
):
    exp_name = "minihack-vae"
    model_name = "deephmm"

    wandb.init(
        mode=wandb_mode,
        project="nesy-mm",
        name=exp_name,
        config={
            "model": model_name,
            "epochs": n_epochs,
            "grid_size": grid_size,
            "horizon": horizon,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "seed": seed,
            "alpha": alpha,
            "beta": beta,
            "dropout": dropout,
            "samples": n_samples,
            "latent_dim": latent_dim,
            "downsample": downsample,
            "discrete_grad_weight": discrete_grad_weight,
            "checkpoint": checkpoint,
        },
    )

    image_shape = (
        (grid_size + 2) * 16 // downsample,
        (grid_size + 2) * 16 // downsample,
        3,
    )

    gamma = 1 / tf.reduce_prod(image_shape)
    gamma = alpha * gamma

    model = MinihackVAEModelHMM(
        grid_size,
        n_samples,
        latent_dim,
        image_shape,
        dropout,
        horizon,
        downsample=downsample,
    )

    print("Loading data...")
    train_data, validation_data, test_data = create_loader(
        grid_size,
        horizon,
        batch_size,
        training_size=5000,
        validation_size=500,
        test_size=1000,
        gen_seed=0,  # we don't want to generate a new dataset for every seed
        load_seed=seed,
        downsample=downsample,
        base_dir=base_dir,
    )

    evaluation_functions = {
        "classifier_accuracy": classifier_accuracy,
        "generative_accuracy": generative_accuracy,
        "reconstruction_error": reconstruction_error,
        "encoder_predictions": encoder_predictions,
    }

    loss = MinihackLoss(alpha, beta, gamma)
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    estimator = RLOO(discrete_grad_weight=discrete_grad_weight)

    model_dir = f"models/{exp_name}/{model_name}"
    model_weights_name = (
        f"{model_name}_{exp_name}_{grid_size}x{grid_size}_horizon{horizon}_seed{seed}"
    )
    trainer = MinihackVAETrainer(
        model,
        optimiser,
        loss,
        estimator,
        evaluation_functions=evaluation_functions,
        test_data=test_data,
        save_dir=model_dir,
        model_name=model_weights_name,
        checkpoint_epochs=checkpoint,
    )

    if weights_path is None:
        # Train model
        trainer.train(train_data, n_epochs, validation_data=validation_data)

        model_dir = f"models/{exp_name}/{model_name}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_weights_name = (
            f"{exp_name}_{grid_size}x{grid_size}_horizon{horizon}_final_seed{seed}.h5"
        )
        model.save_weights(os.path.join(model_dir, model_weights_name))
        model.save_weights(os.path.join(wandb.run.dir, model_weights_name))
        print(
            f"Model weights saved to wandb and {os.path.join(model_dir, model_weights_name)}"
        )
    else:
        # Load model and evaluate
        print(f"Loading model weights from {weights_path}...")
        model.load_weights(weights_path)
        evaluations = trainer.evaluation(test_data)
        for name, evaluation in evaluations.items():
            print(f"{name}: {evaluation.result().numpy()}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--discrete_grad_weight", type=float, default=1e-5)
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    parser.add_argument("--weights_path", type=str, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    run(
        args.grid_size,
        args.n_samples,
        args.latent_dim,
        args.dropout,
        args.downsample,
        args.seed,
        args.batch_size,
        args.horizon,
        args.n_epochs,
        args.alpha,
        args.beta,
        args.learning_rate,
        args.discrete_grad_weight,
        args.checkpoint,
        args.base_dir,
        args.wandb_mode,
        args.weights_path,
    )
