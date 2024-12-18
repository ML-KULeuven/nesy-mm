import os
import tensorflow as tf

from argparse import ArgumentParser
from nesy_mm.src.estimators import RLOO
from nesy_mm.experiments.magnets.data.generator import create_loader
from nesy_mm.experiments.magnets.model import *
from nesy_mm.experiments.magnets.loss import (
    MagnetLoss,
    MagnetSupervisedLoss,
    LogMagnetSupervisedLoss,
)
from nesy_mm.experiments.magnets.trainer import MagnetTrainer
from nesy_mm.experiments.magnets.evaluation import *

# import logging

# logging.getLogger("tensorflow").setLevel(logging.ERROR)

# tf.keras.backend.set_floatx("float64")


def run(
    n_objects, horizon, n_samples, seed, batch_size, n_epochs, learning_rate, log_space
):
    # model = MagnetModel(n_objects, horizon, n_samples, log_space)
    # model = DeepHMMMagnetModel(n_objects, horizon, n_samples)
    model = SimpleDeepHMMModel(n_objects, horizon, n_samples, log_space)

    train_data, validation_data, test_data = create_loader(
        horizon,
        batch_size,
        seed,
        training_size=5000,
        validation_size=500,
        test_size=1000,
        equal_batches=True,
    )

    evaluation_functions = {"accuracy": type_accuracy}

    # loss = MagnetLoss()
    loss = MagnetSupervisedLoss()
    # loss = LogMagnetSupervisedLoss()
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    estimator = RLOO(elbo=False)
    trainer = MagnetTrainer(
        model,
        optimiser,
        loss,
        estimator,
        evaluation_functions=evaluation_functions,
        log_its=100,
    )

    trainer.train(train_data, n_epochs, validation_data=validation_data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_objects", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=11)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--log_space", type=int, default=0)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    run(
        args.n_objects,
        args.horizon,
        args.n_samples,
        args.seed,
        args.batch_size,
        args.n_epochs,
        args.learning_rate,
        bool(args.log_space),
    )
