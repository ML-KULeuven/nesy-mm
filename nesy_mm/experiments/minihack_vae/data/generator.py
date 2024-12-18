import os
import random
import numpy as np
import tensorflow as tf
import einops as E

from matplotlib import pyplot as plt
from typing import Tuple
from nle import nethack
from minihack import LevelGenerator
from minihack import MiniHackNavigation


class Mario(MiniHackNavigation):
    def __init__(
        self, *args, size=5, lit=True, random_start=True, start_loc=None, **kwargs
    ):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", size * 20)

        lvl_gen = LevelGenerator(w=size + 1, h=size + 1, lit=lit)
        lvl_gen.fill_terrain("rect", "|", 0, 0, size + 1, size + 1)

        lvl_gen.add_goal_pos(
            (size + 2, size + 2)
        )  # put the goal outside the walls, so it can't be reached

        if random_start:
            x, y = np.random.randint(1, size, 2)
            x, y = int(x), int(y)
            lvl_gen.set_start_pos((x, y))
        else:
            if start_loc is not None:
                lvl_gen.set_start_pos(start_loc)
            else:
                lvl_gen.set_start_pos((1, 1))

        super().__init__(*args, des_file=lvl_gen.get_des(), **kwargs)


class MarioDataset:
    _actions = tuple(nethack.CompassCardinalDirection)
    _max_episode_steps = 1000
    _blank_char = 32
    _pixel_size = 16
    _x_min = _y_min = None

    def __init__(self, grid_size):
        self._grid_size = grid_size
        self._env = None
        self.env_reset()

    def env_reset(self):
        self._env = Mario(
            size=self._grid_size,
            max_episode_steps=self._max_episode_steps,
            actions=self._actions,
            observation_keys=("chars", "pixel"),
            reward_win=0,
            penalty_step=-1,
            penalty_time=-1,
        )

    def get_grid_size(self):
        return self._grid_size

    def generate_sequence(self, length=2, seed=None):
        assert (
            0 < length <= self._max_episode_steps
        ), f"Length must be between 0 and {self._max_episode_steps}"
        if seed is not None:
            self._seed_everything(seed)
        self.env_reset()
        obs = self._env.reset()
        obs_list = []
        actions = []
        locations = []
        for _ in range(length):
            obs_list.append(self.cropped_pixel_observation(obs))
            locations.append(self.get_agent_location(obs))
            action = self._env.action_space.sample()
            obs, reward, done, info = self._env.step(action)
            actions.append(action)

            if done:
                assert False, "This should never happen"
        return obs_list, actions, locations

    # def generate_loop_sequence(self, length=2, seed=None):
    #     assert (
    #         0 < length <= self._max_episode_steps
    #     ), f"Length must be between 0 and {self._max_episode_steps}"
    #     if seed is not None:
    #         self._seed_everything(seed)
    #     loop_seq = False
    #     while not loop_seq:
    #         self.env_reset()
    #         obs = self._env.reset()
    #         obs_list = []
    #         actions = []
    #         done = False
    #         locations = []
    #         for _ in range(length):
    #             if done:
    #                 # this can't be a loop unless the goal is in the same place as the starting location (impossible)
    #                 break
    #             else:
    #                 obs_list.append(self.cropped_pixel_observation(obs))
    #                 locations.append(self.get_agent_location(obs))
    #                 action = self._env.action_space.sample()
    #                 obs, reward, done, info = self._env.step(action)
    #                 actions.append(action)
    #
    #         # check that the starting location is the same as the ending location
    #         if locations[0] == locations[-1]:
    #             loop_seq = True
    #
    #     return obs_list, actions[:-1], locations, 1.0

    # print sequence of pixel observations in a single plot
    def plot_sequence(
        self, pixel_obs, size=5, show=False, path=None, actions=None, locations=None
    ):
        n_images = len(pixel_obs)
        fig = plt.figure(figsize=(size * n_images, size))
        for i, obs in enumerate(pixel_obs):
            ax = fig.add_subplot(1, n_images, i + 1)
            ax.imshow(obs)
            title_str = ""
            title_str += f" {locations[i]} " if locations and i == n_images - 1 else ""
            title_str += (
                f" {self._pretty_print_action(actions[i])} "
                if actions and i < n_images - 1
                else ""
            )
            ax.set_title(title_str, fontsize=40)
            ax.axis("off")

        # remove space between subplots
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        if path:
            plt.savefig(path, bbox_inches="tight", pad_inches=0)

        if show:
            plt.show()

    def cropped_pixel_observation(self, obs):
        if self._x_min is None:
            self._x_min, self._y_min = np.argwhere(
                obs["chars"] != self._blank_char
            ).min(axis=0)
        x_min = self._x_min * self._pixel_size
        y_min = self._y_min * self._pixel_size
        x_max = (self._x_min + self._grid_size + 2) * self._pixel_size
        y_max = (self._y_min + self._grid_size + 2) * self._pixel_size
        return obs["pixel"][x_min:x_max, y_min:y_max]

    def get_agent_location(self, obs):
        agent_loc = np.argwhere(obs["chars"] == ord("@"))
        assert (
            len(agent_loc) == 1
        ), f"Expected 1 agent location, found {len(agent_loc)}."
        x, y = agent_loc[0]
        return x - self._x_min, y - self._y_min

    def _seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        self._env.seed(seed)
        self._env.action_space.np_random.seed(seed)

    def _pretty_print_action(self, action):
        if action is None:
            return ""
        else:
            act = self._actions[action]
            if act == nethack.CompassCardinalDirection.N:
                return "up"
            elif act == nethack.CompassCardinalDirection.S:
                return "down"
            elif act == nethack.CompassCardinalDirection.E:
                return "right"
            elif act == nethack.CompassCardinalDirection.W:
                return "left"
            else:
                raise NotImplementedError(f"Action {action} not implemented")


def create_sample(dataset, n_steps, seed, print_seq=False, save_fig=False):
    sample = dataset.generate_sequence(length=n_steps, seed=seed)

    if print_seq or save_fig:
        n = dataset.get_grid_size()
        fig_name = f"minihack_vae{n}x{n}_h{n_steps}_seed{seed}.png"
        dataset.plot_sequence(
            sample[0],
            size=n,
            show=print_seq,
            path=fig_name if save_fig else None,
            actions=sample[1],
            locations=sample[2],
        )

    return sample


def print_sample(dataset, sample, tf_record_dir, index):
    info_path = f"{tf_record_dir}/sample_{index}"
    dataset.plot_sequence(sample[0], show=False, path=f"{info_path}.png")

    # print the coordinates of the objects and their types in a txt file
    with open(f"{info_path}.txt", "w") as f:
        f.write(f"\nActions:\n{sample[1]}\n\nLocations:\n{sample[2]}")


def create_tf_records(tf_record_dir, split, size, horizon, grid_size, seed_start):
    print(f"Creating TFRecord files for the {split} set...")
    tf_record_file = tf_record_dir + f"/{split}.tfrecord"

    mario_dataset = MarioDataset(grid_size=grid_size)
    with tf.io.TFRecordWriter(tf_record_file) as writer:
        for i in range(size):
            tf.print(f"Generating {i + 1}/{size} {split} sample...")
            images, actions, locations = create_sample(
                mario_dataset, horizon, seed=seed_start + i
            )
            images = tf.stack(images, 0)
            images = images / 255
            actions = tf.stack(actions, 0)
            actions = tf.one_hot(actions, depth=4, axis=-1)
            locations = tf.stack(locations, 0)

            if i < 5:
                print_sample(
                    mario_dataset,
                    (images, actions, locations),
                    tf_record_dir,
                    seed_start + i,
                )

            feature = {
                "images": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[images.numpy().tobytes()])
                ),
                "actions": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[actions.numpy().tobytes()])
                ),
                "locations": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[locations.numpy().tobytes()])
                ),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    print(f"TFRecord files for the {split} set created successfully!")


def create_loader(
    grid_size: int,
    horizon: int,
    batch_size: int,
    training_size: int = 5000,
    validation_size: int = 500,
    test_size: int = 1000,
    base_dir: str = "",
    gen_seed: int = 0,  # this is used only to generate the dataset
    load_seed: int = 0,
    downsample: int = 1,
    location_classifier: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:

    tf_record_dir = f"{base_dir}nesy_mm/experiments/minihack_vae/data/size{grid_size}_steps{horizon}_seed{gen_seed}_record"
    # tf_record_dir = (
    #     f"{base_dir}/data/size{grid_size}_steps{horizon}_seed{gen_seed}_record"
    # )
    if not os.path.exists(tf_record_dir):
        os.makedirs(tf_record_dir)

        create_tf_records(
            tf_record_dir,
            "training",
            training_size,
            horizon,
            grid_size,
            seed_start=gen_seed,
        )
        create_tf_records(
            tf_record_dir,
            "validation",
            validation_size,
            horizon,
            grid_size,
            seed_start=gen_seed + training_size,
        )
        create_tf_records(
            tf_record_dir,
            "test",
            test_size,
            horizon,
            grid_size,
            seed_start=gen_seed + training_size + validation_size,
        )

    feature_description = {
        "images": tf.io.FixedLenFeature([], tf.string),
        "actions": tf.io.FixedLenFeature([], tf.string),
        "locations": tf.io.FixedLenFeature([], tf.string),
    }

    test_batch_size = 50

    image_shape = (
        batch_size,
        horizon,
        (grid_size + 2) * 16,
        (grid_size + 2) * 16,
        3,
    )
    test_image_shape = (
        test_batch_size,
        horizon,
        (grid_size + 2) * 16,
        (grid_size + 2) * 16,
        3,
    )

    if location_classifier:

        def _parse_function(example_proto):
            features = tf.io.parse_example(example_proto, feature_description)
            images = tf.io.decode_raw(features["images"], tf.float64)
            images = tf.cast(images, tf.float32)
            images = tf.reshape(images, image_shape)
            images = E.reduce(
                images,
                "b t (h r1) (w r2) c -> b t h w c",
                "mean",
                r1=downsample,
                r2=downsample,
            )
            images = E.rearrange(images, "b t h w c -> (b t) h w c")
            locations = tf.io.decode_raw(features["locations"], tf.int32)
            locations = tf.reshape(locations, (batch_size * horizon, 2))
            locations = tf.one_hot(locations, depth=grid_size + 2, axis=-1)
            return images, locations

        def _parse_function_test(example_proto):
            features = tf.io.parse_example(example_proto, feature_description)
            images = tf.io.decode_raw(features["images"], tf.float64)
            images = tf.cast(images, tf.float32)
            images = tf.reshape(images, test_image_shape)
            images = E.reduce(
                images,
                "b t (h r1) (w r2) c -> b t h w c",
                "mean",
                r1=downsample,
                r2=downsample,
            )
            images = E.rearrange(images, "b t h w c -> (b t) h w c")
            locations = tf.io.decode_raw(features["locations"], tf.int32)
            locations = tf.reshape(locations, (test_batch_size * horizon, 2))
            locations = tf.one_hot(locations, depth=grid_size + 2, axis=-1)
            return images, locations

    else:

        def _parse_function(example_proto):
            features = tf.io.parse_example(example_proto, feature_description)
            images = tf.io.decode_raw(features["images"], tf.float64)
            images = tf.cast(images, tf.float32)
            images = tf.reshape(images, image_shape)
            images = E.reduce(
                images,
                "b t (h r1) (w r2) c -> b t h w c",
                "mean",
                r1=downsample,
                r2=downsample,
            )
            actions = tf.io.decode_raw(features["actions"], tf.float32)
            actions = tf.reshape(actions, (batch_size, horizon, 4))
            actions = tf.argmax(actions, axis=-1)
            locations = tf.io.decode_raw(features["locations"], tf.int32)
            locations = tf.reshape(locations, (batch_size, horizon, 2))

            return images, actions, locations[:, -1, :]

        def _parse_function_test(example_proto):
            features = tf.io.parse_example(example_proto, feature_description)
            images = tf.io.decode_raw(features["images"], tf.float64)
            images = tf.cast(images, tf.float32)
            images = tf.reshape(images, test_image_shape)
            images = E.reduce(
                images,
                "b t (h r1) (w r2) c -> b t h w c",
                "mean",
                r1=downsample,
                r2=downsample,
            )
            actions = tf.io.decode_raw(features["actions"], tf.float32)
            actions = tf.reshape(actions, (test_batch_size, horizon, 4))
            actions = tf.argmax(actions, axis=-1)
            locations = tf.io.decode_raw(features["locations"], tf.int32)
            locations = tf.reshape(locations, (test_batch_size, horizon, 2))
            return images, actions, locations

    train_dataset = tf.data.TFRecordDataset(f"{tf_record_dir}/training.tfrecord")
    train_dataset = train_dataset.shuffle(training_size // 10, seed=load_seed)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(
        _parse_function, num_parallel_calls=tf.data.AUTOTUNE
    )

    validation_dataset = tf.data.TFRecordDataset(f"{tf_record_dir}/validation.tfrecord")
    validation_dataset = validation_dataset.shuffle(
        validation_size // 5, seed=load_seed
    )
    validation_dataset = validation_dataset.batch(test_batch_size)
    validation_dataset = validation_dataset.map(
        _parse_function_test, num_parallel_calls=tf.data.AUTOTUNE
    )

    test_dataset = tf.data.TFRecordDataset(f"{tf_record_dir}/test.tfrecord")
    test_dataset = test_dataset.shuffle(test_size // 5, seed=load_seed)
    test_dataset = test_dataset.batch(test_batch_size)
    test_dataset = test_dataset.map(
        _parse_function_test, num_parallel_calls=tf.data.AUTOTUNE
    )

    return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    ds = MarioDataset(5)

    n = 1
    entries = []
    for i in range(n):
        print(f"Generating {i + 1}/{n} sample...")
        entry = create_sample(ds, 4, seed=i, print_seq=True, save_fig=True)
        entries.append(entry)
