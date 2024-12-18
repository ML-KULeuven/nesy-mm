import gym
import minihack
import pickle
import os
import random
import numpy as np
import tensorflow as tf
from typing import Tuple

from matplotlib import pyplot as plt
from nle import nethack
from minihack import LevelGenerator
from minihack import MiniHackNavigation
from minihack.tiles.glyph_mapper import GlyphMapper


class RoomWithEnemies(MiniHackNavigation):
    def __init__(self, *args, size=5, n_enemies=1, lit=True, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", size * 20)

        kwargs["character"] = kwargs.pop("character", "rog-hum-cha-mal")

        lvl_gen = LevelGenerator(w=size + 1, h=size + 1, lit=lit)
        lvl_gen.fill_terrain("rect", "|", 0, 0, size + 1, size + 1)
        lvl_gen.add_goal_pos(
            (size + 2, size + 2)
        )  # put the goal outside the walls, so it can't be reached
        x, y = np.random.randint(1, size, 2)
        x_start, y_start = int(x), int(y)
        lvl_gen.set_start_pos((x_start, y_start))
        self.player_start = (x_start, y_start)
        self.enemies_start = []
        enemies_deployed = 0
        while enemies_deployed < n_enemies:
            x, y = np.random.randint(1, size, 2)
            x, y = int(x), int(y)
            if (x, y) != (x_start, y_start) and (x, y) not in self.enemies_start:
                """An uruk has a 10 + 1 + 5 - 1 / 20 = 0.75 chance of hitting the player with 1d8"""  # TODO: source?
                # A wolf has 10 + 7 (agent AC) + 5 (level of the wolf) = 22 / 20 > 1 chance of hitting the player
                lvl_gen.add_monster(
                    name="imp",  # "Uruk-hai" is spawned with ranged weapons in some cases
                    symbol="i",
                    place=(x, y),
                    args=("hostile", "awake"),
                )
                enemies_deployed += 1
                self.enemies_start.append((x, y))

        super().__init__(*args, des_file=lvl_gen.get_des(), **kwargs)


class RoomWithEnemiesDataset:
    # Info on blstats: https://gitlab.aicrowd.com/debjoy_saha/neurips-2021-the-nethack-challenge/-/blob/master/notebooks/NetHackTutorial.ipynb
    _actions = tuple(nethack.CompassCardinalDirection)
    _max_episode_steps = 1000
    _blank_glyph = 2359
    _agent_glyph = 337  # rogue (337) barbarian (328)
    _enemy_glyph = 51  # orc (71), wolf (20), imp (51)
    _pixel_size = 16
    _x_min = _y_min = None

    def __init__(self, grid_size, n_enemies):
        self._grid_size = grid_size
        self._n_enemies = n_enemies
        self._env = None
        self.env_reset()

    def get_grid_size(self):
        return self._grid_size

    def env_reset(self):
        self._env = RoomWithEnemies(
            size=self._grid_size,
            n_enemies=self._n_enemies,
            max_episode_steps=self._max_episode_steps,
            actions=self._actions,
            observation_keys=("blstats", "glyphs"),
            reward_win=0,
            penalty_step=-1,
            penalty_time=-1,
        )

    def generate_sequence(self, length=2, seed=None):
        assert (
            0 < length <= self._max_episode_steps
        ), f"Length must be between 0 and {self._max_episode_steps}"
        if seed is not None:
            self._seed_everything(seed)
        self.env_reset()
        obs = self._env.reset()
        glyphs_obs_list = [self.cropped_observation(obs)]
        monster_hits = [0]
        agent_hps = [obs["blstats"][10]]
        actions = []
        done = False
        agent_dead = False
        locations = [self.get_agent_location(obs)]
        enemies_locations = [self.get_enemies_locations(obs)]
        enemy_dead = False
        ranged_attack = False
        for t in range(length - 1):
            if done:
                # finish filling in the sequence with the last (blank) observations and action 0, which means "up"
                glyphs_obs_list.append(glyphs_obs_list[-1])
                actions.append(0)
                monster_hits.append(0)
                agent_hps.append(agent_hps[-1])
                locations.append(locations[-1])
                enemies_locations.append(enemies_locations[-1])
            else:
                action = self._env.action_space.sample()
                actions.append(action)

                obs, reward, done, info = self._env.step(action)  # execute step

                # check if agent is dead and update its location
                if (
                    info["end_status"] == 1
                ):  # 1 is DEATH, see StepStatus in NetHackStaircase class in nle.tasks
                    agent_dead = True
                    locations.append(locations[-1])
                else:
                    locations.append(self.get_agent_location(obs))

                # check if an enemy is dead and update enemies locations
                enemies_locs = self.get_enemies_locations(obs)
                if len(enemies_locs) < self._n_enemies:
                    enemy_dead = (
                        not agent_dead
                    )  # enemy dead only if it's not in the room and the agent is still alive
                    enemies_locations.append(enemies_locations[-1])
                else:
                    enemies_locations.append(enemies_locs)

                # check if the enemy hit the agent
                if (
                    agent_hps[-1] > obs["blstats"][10]
                ):  # 10th index is HP (11th is max HP)
                    for e in range(self._n_enemies):
                        if (
                            abs(locations[-1][0] - enemies_locations[-1][e][0]) > 1
                            or abs(locations[-1][1] - enemies_locations[-1][e][1]) > 1
                        ):
                            # print(f"{t}) Enemy id{e}@({enemies_locations[-1][e][0]},{enemies_locations[-1][e][1]}) "
                            #       f"hit the agent@({locations[-1][0]},{locations[-1][1]}), "
                            #       f"but the distance between them is greater than 1.")
                            ranged_attack = True
                    monster_hits.append(1)
                else:
                    monster_hits.append(0)

                # update stats
                agent_hps.append(obs["blstats"][10])
                glyphs_obs_list.append(self.cropped_observation(obs))

        return {
            "actions": actions,
            "glyphs": glyphs_obs_list,
            "player_locs": locations,
            "enemies_locs": enemies_locations,
            "agent_hps": agent_hps,
            "enemy_hits": monster_hits,
            "dead": tf.constant(agent_dead, dtype=tf.bool),
            "enemy_dead": enemy_dead,
            "ranged_attack": ranged_attack,
        }

    # print sequence of pixel observations in a single plot
    def plot_sequence(
        self, glyphs_obs, actions, locations, save_fig=False, fig_name=None
    ):
        pixel_obs = [self.get_pixel_from_glyphs(glyphs) for glyphs in glyphs_obs]
        fig_width = self._grid_size * len(pixel_obs)
        fig_height = self._grid_size + 1
        fig = plt.figure(figsize=(fig_width, fig_height))
        for i, obs in enumerate(pixel_obs):
            fig.add_subplot(1, len(pixel_obs), i + 1)
            plt.imshow(obs)
            # set action as subplot title
            title_str = f"{locations[i]}, " if i == 0 else ""
            if i < len(actions):
                title_str += f"{self._pretty_print_action(actions[i])}"
            plt.title(title_str, fontsize=40)
        # remove axis
        for ax in fig.get_axes():
            ax.axis("off")
        # remove space between subplots
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        if save_fig:
            fig_name = fig_name or f"dataset_{len(glyphs_obs)}.png"
            plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)
        plt.show()

    def cropped_observation(self, obs):
        if self._x_min is None:
            self._x_min, self._y_min = np.argwhere(
                obs["glyphs"] != self._blank_glyph
            ).min(axis=0)
        x_min = self._x_min
        y_min = self._y_min
        x_max = self._x_min + self._grid_size + 2
        y_max = self._y_min + self._grid_size + 2
        return obs["glyphs"][x_min:x_max, y_min:y_max].copy()

    @staticmethod
    def get_pixel_from_glyphs(glyphs):
        mapper = GlyphMapper()
        img = mapper.to_rgb(glyphs)
        return img

    def get_agent_location(self, obs):
        agent_loc = np.argwhere(obs["glyphs"] == self._agent_glyph)
        assert (
            len(agent_loc) == 1
        ), f"Expected 1 agent location, found {len(agent_loc)}."
        return agent_loc[0][0] - self._x_min, agent_loc[0][1] - self._y_min

    def get_enemies_locations(self, obs):
        enemies_loc = np.argwhere(obs["glyphs"] == self._enemy_glyph)
        assert (
            len(enemies_loc) <= self._n_enemies
        ), f"Expected {self._n_enemies} enemies locations, found {len(enemies_loc)}."
        return [(x - self._x_min, y - self._y_min) for x, y in enemies_loc]

    def _seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        self._env.seed(seed)
        self._env.action_space.np_random.seed(seed)

    def _pretty_print_action(self, action):
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


def create_loader(
    grid_size: int,
    horizon: int,
    num_enemies: int,
    batch_size: int,
    training_size: int = 5000,
    validation_size: int = 500,
    test_size: int = 1000,
    gen_seed: int = 0,
    load_seed: int = 0,
    base_dir: str = "",
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:

    tf_record_dir = f"{base_dir}nesy_mm/experiments/minihack_transition/data/size{grid_size}_steps{horizon}_seed{gen_seed}_enemies{num_enemies}_record"
    if not os.path.exists(tf_record_dir):
        os.makedirs(tf_record_dir)

        create_tf_records(
            tf_record_dir,
            "training",
            training_size,
            horizon,
            grid_size,
            num_enemies,
            seed_start=gen_seed,
        )
        create_tf_records(
            tf_record_dir,
            "validation",
            validation_size,
            horizon,
            grid_size,
            num_enemies,
            seed_start=training_size,
        )
        create_tf_records(
            tf_record_dir,
            "test",
            test_size,
            horizon,
            grid_size,
            num_enemies,
            seed_start=training_size + validation_size,
        )

    feature_description = {
        "actions": tf.io.FixedLenFeature([], tf.string),
        "glyphs": tf.io.FixedLenFeature([], tf.string),
        "player_locs": tf.io.FixedLenFeature([], tf.string),
        "enemies_locs": tf.io.FixedLenFeature([], tf.string),
        "agent_hps": tf.io.FixedLenFeature([], tf.string),
        "enemy_hits": tf.io.FixedLenFeature([], tf.string),
        "dead": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_function(example_proto):
        features = tf.io.parse_example(example_proto, feature_description)
        actions = tf.io.decode_raw(features["actions"], tf.float32)
        actions = tf.reshape(actions, (batch_size, horizon - 1, 4))
        # glyphs = tf.io.decode_raw(features["glyphs"], tf.int32)
        # glyphs = tf.reshape(glyphs, (batch_size, horizon, grid_size+2, grid_size+2))
        player_locs = tf.io.decode_raw(features["player_locs"], tf.int32)
        player_locs = tf.reshape(player_locs, (batch_size, horizon, 2))
        # enemies_locs = tf.io.decode_raw(features["enemies_locs"], tf.int32)
        # enemies_locs = tf.reshape(enemies_locs, (batch_size, horizon, num_enemies, 2))
        # agent_hps = tf.io.decode_raw(features["agent_hps"], tf.int32)
        # agent_hps = tf.reshape(agent_hps, (batch_size, horizon))
        enemy_hits = tf.io.decode_raw(features["enemy_hits"], tf.int32)
        enemy_hits = tf.reshape(enemy_hits, (batch_size, horizon))
        dead = tf.io.decode_raw(features["dead"], tf.bool)
        dead = tf.cast(dead, tf.int32)
        return (
            actions,
            # glyphs,
            player_locs[:, 0],
            # enemies_locs,
            # agent_hps,
            enemy_hits,
            dead,
        )

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
    validation_dataset = validation_dataset.batch(batch_size)
    validation_dataset = validation_dataset.map(
        _parse_function, num_parallel_calls=tf.data.AUTOTUNE
    )

    test_dataset = tf.data.TFRecordDataset(f"{tf_record_dir}/test.tfrecord")
    test_dataset = test_dataset.shuffle(test_size // 5, seed=load_seed)
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.map(
        _parse_function, num_parallel_calls=tf.data.AUTOTUNE
    )

    return train_dataset, validation_dataset, test_dataset


def create_tf_records(
    tf_record_dir, split, size, horizon, grid_size, num_enemies, seed_start
):
    print(f"Creating TFRecord files for the {split} set...")
    tf_record_file = tf_record_dir + f"/{split}.tfrecord"

    enemyroom_dataset = RoomWithEnemiesDataset(grid_size, num_enemies)
    with tf.io.TFRecordWriter(tf_record_file) as writer:
        for i in range(size):
            tf.print(f"Generating {i + 1}/{size} {split} sample...")
            sample = create_sample(enemyroom_dataset, horizon, seed=seed_start + i)
            actions = sample["actions"]
            actions = tf.stack(actions, 0)
            actions = tf.one_hot(actions, depth=4, axis=-1)
            glyphs = sample["glyphs"]
            glyphs = tf.stack(glyphs, 0)
            player_locs = sample["player_locs"]
            player_locs = tf.stack(player_locs, 0)
            enemies_locs = sample["enemies_locs"]
            enemies_locs = tf.stack(enemies_locs, 0)
            agent_hps = sample["agent_hps"]
            agent_hps = tf.stack(agent_hps, 0)
            enemy_hits = sample["enemy_hits"]
            enemy_hits = tf.stack(enemy_hits, 0)
            dead = sample["dead"]

            feature = {
                "actions": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[actions.numpy().tobytes()])
                ),
                "glyphs": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[glyphs.numpy().tobytes()])
                ),
                "player_locs": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[player_locs.numpy().tobytes()])
                ),
                "enemies_locs": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[enemies_locs.numpy().tobytes()]
                    )
                ),
                "agent_hps": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[agent_hps.numpy().tobytes()])
                ),
                "enemy_hits": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[enemy_hits.numpy().tobytes()])
                ),
                "dead": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[dead.numpy().tobytes()])
                ),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    print(f"TFRecord files for the {split} set created successfully!")


def create_sample(dataset, n_steps, seed, print_seq=False, save_fig=False):
    # exclude sequences where enemies die before the player, and also when there are ranged attacks
    while True:
        seq = dataset.generate_sequence(length=n_steps, seed=seed)
        if not seq["enemy_dead"]:
            break
        # TODO: fix the ranged attack check to work with multiple enemies
        # elif not seq["ranged_attack"]:
        #     break
        # print("New sequence...")

    if print_seq or save_fig:
        n = dataset.get_grid_size()
        fig_name = f"enemyroom{n}x{n}_h{n_steps}_seed{seed}.png"
        dataset.plot_sequence(
            seq["glyphs"],
            seq["actions"],
            seq["player_locs"],
            save_fig=save_fig,
            fig_name=fig_name,
        )

    return seq


if __name__ == "__main__":
    # (grid_size, n_enemies)
    ds = RoomWithEnemiesDataset(5, 2)

    n = 1
    entries = []
    for i in range(n):
        print(f"Generating {i + 1}/{n} sample...")
        entry = create_sample(ds, 4, seed=i, print_seq=True, save_fig=True)
        entries.append(entry)

    # count how many times the player died
    dead = sum([int(entry["dead"]) for entry in entries])
    print(f"Player died {dead}/{n} times.")
