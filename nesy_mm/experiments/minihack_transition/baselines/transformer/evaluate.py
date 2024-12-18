import os
import yaml
import tensorflow as tf

from keras.losses import BinaryCrossentropy

from argparse import ArgumentParser
from nesy_mm.experiments.minihack_transition.data.generator import create_loader
from nesy_mm.experiments.minihack_transition.baselines.transformer.model import Mirage
from nesy_mm.experiments.minihack_transition.baselines.transformer.trainer import (
    TransformerTrainer,
)
from nesy_mm.experiments.minihack_transition.baselines.transformer.evaluation import *


def evaluate(
    grid_size,
    horizon,
    n_enemies,
    test_grid_size,
    test_horizon,
    n_test_enemies,
    embedding_size,
    base_dir,
    batch_size,
    seed,
    n_epochs,
):

    exp_name = "minihack-transition"
    model_name = "transformer"

    model = Mirage(grid_size, embedding_size)

    model_dir = f"models/{exp_name}/{model_name}"
    model_weights_name = f"{model_name}_{exp_name}_{grid_size}x{grid_size}_horizon{horizon}_enemies{n_enemies}_seed{seed}"

    print("Loading Test Data...")
    train_data, validation_data, test_data = create_loader(
        test_grid_size,
        test_horizon,
        n_test_enemies,
        batch_size,
        training_size=5000,
        validation_size=500,
        test_size=1000,
        gen_seed=0,  # we don't want to generate a new dataset for every seed
        load_seed=seed + 1,
        base_dir=base_dir,
    )

    del train_data
    del validation_data

    evaluation_functions = {
        "balanced_accuracy": balanced_accuracy,
        "f1_score": f1_score,
        "positive_accuracy": positive_accuracy,
        "negative_accuracy": negative_accuracy,
    }

    loss = BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    trainer = TransformerTrainer(
        model,
        optimizer,
        loss,
        evaluation_functions=evaluation_functions,
        test_data=test_data,
        save_dir=model_dir,
        model_name=model_weights_name,
        checkpoint_epochs=None,
        wandb_on=False,
    )

    model_dir = f"models/{exp_name}/{model_name}"
    if n_epochs > 0:
        model_weights_name = f"{model_name}_{exp_name}_{grid_size}x{grid_size}_horizon{horizon}_enemies{n_enemies}_seed{seed}_checkpoint{n_epochs}.h5"
    else:
        model_weights_name = f"{model_name}_{exp_name}_{grid_size}x{grid_size}_horizon{horizon}_enemies{n_enemies}_seed{seed}"
    model.load_weights(f"{model_dir}/{model_weights_name}")

    evaluations = trainer.evaluation(test_data)

    logging_dict = []
    for name, evaluation in evaluations.items():
        print(f"{name}: {evaluation.result().numpy()}")
        logging_dict += [("test_" + name, float(evaluation.result().numpy()))]

    if not os.path.exists(f"evaluation/{model_weights_name}"):
        os.makedirs(f"evaluation/{model_weights_name}")

    with open(
        f"evaluation/{model_weights_name}/grid{test_grid_size}_horizon{test_horizon}_enemies{n_test_enemies}.yaml",
        "w",
    ) as file:
        yaml.dump(logging_dict, file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--n_enemies", type=int, default=1)
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=-1)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    configurations = []
    configurations += [(10, 1, 10)]
    configurations += [(10, 2, 10)]
    configurations += [(20, 1, 10)]
    configurations += [(20, 2, 10)]
    configurations += [(10, 1, 15)]
    configurations += [(10, 2, 15)]
    configurations += [(20, 1, 15)]
    configurations += [(20, 2, 15)]

    for test_horizon, n_test_enemies, test_grid_size in configurations:
        evaluate(
            args.grid_size,
            args.horizon,
            args.n_enemies,
            test_grid_size,
            test_horizon,
            n_test_enemies,
            args.embedding_size,
            args.base_dir,
            args.batch_size,
            args.seed,
            args.n_epochs,
        )
