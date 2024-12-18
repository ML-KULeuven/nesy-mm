import os
import wandb
import tensorflow as tf

from argparse import ArgumentParser
from keras.losses import BinaryCrossentropy

from nesy_mm.experiments.minihack_transition.data.generator import create_loader

from nesy_mm.experiments.minihack_transition.baselines.transformer.model import Mirage
from nesy_mm.experiments.minihack_transition.baselines.transformer.trainer import (
    TransformerTrainer,
)
from nesy_mm.experiments.minihack_transition.baselines.transformer.evaluation import *


def run(
    grid_size,
    horizon,
    n_enemies,
    learning_rate,
    embedding_size,
    seed,
    batch_size,
    n_epochs,
    base_dir,
    wandb_mode,
    log_its=100,
):

    exp_name = "minihack-transition"
    model_name = "transformer"

    wandb.init(
        mode=wandb_mode,
        project="nesy-mm",
        name=exp_name,
        config={
            "model": model_name,
            "epochs": n_epochs,
            "grid_size": grid_size,
            "horizon": horizon,
            "n_enemies": n_enemies,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "seed": seed,
        },
    )

    model = Mirage(grid_size, embedding_size)

    model_dir = f"models/{exp_name}/{model_name}"
    model_weights_name = f"{model_name}_{exp_name}_{grid_size}x{grid_size}_horizon{horizon}_enemies{n_enemies}_seed{seed}"

    print("Loading data...")
    train_data, validation_data, test_data = create_loader(
        grid_size,
        horizon,
        n_enemies,
        batch_size,
        training_size=5000,
        validation_size=500,
        test_size=1000,
        gen_seed=0,  # we don't want to generate a new dataset for every seed
        load_seed=seed,
        base_dir=base_dir,
    )

    evaluation_functions = {
        "balanced_accuracy": balanced_accuracy,
        "f1_score": f1_score,
        "positive_accuracy": positive_accuracy,
        "negative_accuracy": negative_accuracy,
    }

    loss = BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    trainer = TransformerTrainer(
        model,
        optimizer,
        loss,
        evaluation_functions=evaluation_functions,
        test_data=validation_data,
        save_dir=model_dir,
        model_name=model_weights_name,
        checkpoint_epochs=10,
        log_its=log_its,
    )

    trainer.train(train_data, n_epochs, validation_data=validation_data)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_weights(f"{model_dir}/{model_weights_name}")

    evaluations = trainer.evaluation(test_data)

    logging_dict = dict()
    for name, evaluation in evaluations.items():
        print(f"{name}: {evaluation.result().numpy()}")
        logging_dict["test_" + name] = float(evaluation.result().numpy())

    wandb.log(logging_dict)
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--n_enemies", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    parser.add_argument("--log_its", type=int, default=100)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    run(
        args.grid_size,
        args.horizon,
        args.n_enemies,
        args.learning_rate,
        args.embedding_size,
        args.seed,
        args.batch_size,
        args.n_epochs,
        base_dir=args.base_dir,
        wandb_mode=args.wandb_mode,
        log_its=args.log_its,
    )
