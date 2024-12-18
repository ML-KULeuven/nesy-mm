import os
import yaml
import tensorflow as tf

from argparse import ArgumentParser
from nesy_mm.src.estimators import RLOO
from nesy_mm.experiments.minihack_transition.evaluation import *
from nesy_mm.experiments.minihack_transition.model import LearnEnemyAction
from nesy_mm.experiments.minihack_transition.trainer import MinihackTransitionTrainer
from nesy_mm.experiments.minihack_transition.data.generator import create_loader
from nesy_mm.experiments.minihack_transition.loss import DeadOrNot


def evaluate(
    n_samples,
    grid_size,
    horizon,
    n_enemies,
    test_grid_size,
    test_horizon,
    n_test_enemies,
    relational,
    base_dir,
    batch_size,
    seed,
    n_epochs,
):

    exp_name = "minihack-transition"
    model_name = "nesymm"

    relational = bool(relational)

    model = LearnEnemyAction(grid_size, n_samples, relational=relational)

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

    loss = DeadOrNot()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    estimator = RLOO(discrete_grad_weight=1.0)

    model_dir = f"models/{exp_name}/{model_name}"
    if n_epochs > 0:
        model_weights_name = f"{model_name}_{exp_name}_{grid_size}x{grid_size}_horizon{horizon}_enemies{n_enemies}_seed{seed}_checkpoint{n_epochs}.h5"
    else:
        model_weights_name = f"{model_name}_{exp_name}_{grid_size}x{grid_size}_horizon{horizon}_enemies{n_enemies}_seed{seed}"
    model.load_weights(f"{model_dir}/{model_weights_name}")

    print("Hit chance: ", float(tf.nn.sigmoid(model.hit.hit_chance).numpy()))

    trainer = MinihackTransitionTrainer(
        model,
        optimizer,
        loss,
        estimator,
        evaluation_functions=evaluation_functions,
        horizon=horizon,
        test_horizon=test_horizon,
        n_enemies=n_enemies,
        n_test_enemies=n_test_enemies,
        test_data=test_data,
        grid_size=grid_size,
        test_grid_size=test_grid_size,
        save_dir=model_dir,
        model_name=model_weights_name,
        checkpoint_epochs=None,
        wandb_on=False,
    )

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
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--n_enemies", type=int, default=1)
    parser.add_argument("--relational", type=int, default=1)
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
            args.n_samples,
            args.grid_size,
            args.horizon,
            args.n_enemies,
            test_grid_size,
            test_horizon,
            n_test_enemies,
            args.relational,
            args.base_dir,
            args.batch_size,
            args.seed,
            args.n_epochs,
        )
