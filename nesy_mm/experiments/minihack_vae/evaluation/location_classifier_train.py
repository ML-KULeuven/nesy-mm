import os
import numpy as np
import tensorflow as tf
import einops as E
from nle import nethack

from nesy_mm.experiments.minihack_vae.neural.classifier import AgentClassifier
from nesy_mm.experiments.minihack_vae.data.generator import create_loader
from nesy_mm.experiments.minihack_vae.data.generator import Mario

blank = 32
size_pixel = 16

def get_crop_pixel_from_observation(observation):
    coords = np.argwhere(observation["chars"] != blank)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    non_empty_pixels = observation["pixel"][
        x_min * size_pixel: (x_max + 1) * size_pixel,
        y_min * size_pixel: (y_max + 1) * size_pixel,
    ]
    return non_empty_pixels


def test_classifier(location_classifier, size, downsample):
    # test location_classifier
    env = Mario(
        size=size,
        max_episode_steps=1000,
        actions=tuple(nethack.CompassCardinalDirection),
        observation_keys=("chars", "pixel"),
        random_start=False,
        start_loc=(2, 5),
        reward_win=0,
        penalty_step=-1,
        penalty_time=-1,
    )
    obs = env.reset()
    img = get_crop_pixel_from_observation(obs)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = E.reduce(img, "(w r1) (h r2) c -> w h c", "mean", r1=downsample, r2=downsample)
    img = E.rearrange(img, "w h c -> 1 w h c")

    location = location_classifier(img)
    # location = tf.nn.softmax(location)
    location = tf.argmax(location, axis=-1)
    print(location.numpy())


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--downsample", type=int, default=1)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    downsample = args.downsample
    grid_size = args.grid_size
    base_dir = args.base_dir
    seed = args.seed
    horizon = 10 if grid_size == 5 else 25
    batch_size = 5  # this is going to be multiplied by the horizon
    image_shape = (
        (grid_size + 2) * size_pixel // downsample,
        (grid_size + 2) * size_pixel // downsample,
        3,
    )
    classifier = AgentClassifier(grid_size, image_shape)

    downsample_str = f"_downsample{downsample}" if downsample > 1 else ""
    model_weights_name = f"location_classifier{grid_size}x{grid_size}{downsample_str}_weights.h5"

    if not args.test_only:
        # Load Data
        print("Loading Data...")
        train_data, val_data, test_data = create_loader(
            grid_size,
            horizon,
            batch_size,
            training_size=1000,
            validation_size=100 if grid_size == 5 else 200,
            test_size=100 if grid_size == 5 else 200,
            base_dir=base_dir,
            gen_seed=seed,
            downsample=downsample,
            location_classifier=True,
        )

        # Training
        print("Training...")
        classifier.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        classifier.fit(
            train_data,
            epochs=10,
            validation_data=val_data,
        )

        print("Evaluation...")
        classifier.evaluate(test_data)
        print("Done.")

        # Save model weights
        classifier.save_weights(
            os.path.join(f"nesy_mm/experiments/minihack_vae/evaluation/", model_weights_name)
        )

    print("Testing the classifier...")
    classifier = AgentClassifier(grid_size, image_shape)
    classifier.build((None,) + image_shape)
    classifier.load_weights(os.path.join(f"nesy_mm/experiments/minihack_vae/evaluation/", model_weights_name))
    test_classifier(classifier, grid_size, downsample)
    print("Done.")
