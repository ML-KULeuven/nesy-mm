# This generator simulates a simple environment with magnets and ferromagnetic and non-ferromagnetic objects.
# The environment is initialized with a given number of each type of object, and the objects are placed randomly in the
# environment at a distance from each other greater than a given interaction radius. In this way the objects will
# interact only if we act on the environment.
# The only possible actions are: reset or move an object towards another object.
# If in the movement of an object it collides with another object, the environment stops and returns a done signal.
# The orientation of the magnets is decided by flipping a coin.
import os
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import einops as E

from typing import Tuple
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# enum with three classes of objects
class ObjectType:
    MAGNET = 0
    FERROMAGNETIC = 1
    NON_FERROMAGNETIC = 2


class InteractionType:
    ATTRACTION = 0
    REPULSION = 1


class MagnetsEnv:
    _obj_radius = 0.02
    _step_size = 0.1
    _interaction_radius = (
        _step_size * 3
    )  # Note: Must be greater than _step_size to keep objects from overlapping
    _magnetic_force_const = 0.05
    _done = False
    _debug = False

    def __init__(
        self, n_magnets, n_ferromagnetic, n_non_ferromagnetic, seed=None, debug=False
    ):
        self.n_magnets = n_magnets
        self.n_ferromagnetic = n_ferromagnetic
        self.n_non_ferromagnetic = n_non_ferromagnetic
        self._total_objects = n_magnets + n_ferromagnetic + n_non_ferromagnetic
        self._seed = seed
        self._debug = debug

        if self._total_objects < 2:
            raise ValueError("The environment must have at least two objects.")

        if self._seed is not None:
            np.random.seed(self._seed)

        self._objects_coord = {}
        self._objects_coord_init = {}
        self._objects_type = {}

    def move_object(self, obj_coord, target_coord, force=None):
        # moves obj towards target of a distance of _step_size
        direction = np.array(
            [target_coord[0] - obj_coord[0], target_coord[1] - obj_coord[1]]
        )
        direction_norm = direction / np.linalg.norm(direction)
        force = 1 if force is None else force
        movement = direction_norm * self._step_size * force
        # if the movement is greater than the distance to the target, move the object to the touch the target
        delta_x = direction[0]
        delta_y = direction[1]
        angle = math.atan2(delta_x, delta_y)
        diameter_delta = np.array(
            [
                math.sin(angle) * self._obj_radius * 2,
                math.cos(angle) * self._obj_radius * 2,
            ]
        )
        safe_distance = direction - diameter_delta
        if np.linalg.norm(movement) > np.linalg.norm(safe_distance):
            movement = safe_distance

        new_obj_x = obj_coord[0] + movement[0]
        new_obj_y = obj_coord[1] + movement[1]

        # don't move the object beyond the bounds of the environment
        new_obj_x = max(0 + self._obj_radius, min(1 - self._obj_radius, new_obj_x))
        new_obj_y = max(0 + self._obj_radius, min(1 - self._obj_radius, new_obj_y))

        return new_obj_x, new_obj_y

    def step(self, held_object, target_object):
        if not self._done:
            # move the object towards the target object of a distance of _step_size
            new_held = self.move_object(
                self._objects_coord[held_object], self._objects_coord[target_object]
            )
            self._objects_coord[held_object] = new_held

            # check all the objects except the held one to see if they are interacting
            for obj in self._objects_coord:
                if obj != held_object:
                    obj_coord = self._objects_coord[obj]
                    if self.interacting(
                        new_held,
                        obj_coord,
                        self._objects_type[held_object],
                        self._objects_type[obj],
                    ):
                        # understand if the objects are attracting or repelling
                        interaction = InteractionType.ATTRACTION
                        if (
                            self._objects_type[held_object] == ObjectType.MAGNET
                            and self._objects_type[obj] == ObjectType.MAGNET
                        ):
                            # flip a coin to decide the orientation of the magnets
                            # note: this means that at every time step a magnet can change its orientation
                            # this simulates the fact that the held object can be rotated by the user
                            interaction = np.random.choice(
                                [InteractionType.ATTRACTION, InteractionType.REPULSION]
                            )

                        # compute the force between the two objects and move them accordingly...
                        diameter = self._obj_radius * 2
                        dist_from_centers = max(
                            self.distance(new_held, obj_coord) + diameter, diameter
                        )
                        force = self._magnetic_force_const * (dist_from_centers**-2)

                        # ...if they don't overlap already
                        if not self.overlapping(new_held, obj_coord):
                            if interaction == InteractionType.ATTRACTION:
                                # move obj towards the held object with magnetic force (unless they already overlap)
                                self._objects_coord[obj] = self.move_object(
                                    obj_coord, new_held, force
                                )
                            else:
                                # move obj away from the held object with magnetic force
                                extreme_x_movement = (obj_coord[0] - new_held[0]) * (
                                    1 / self._obj_radius
                                )
                                extreme_y_movement = (obj_coord[1] - new_held[1]) * (
                                    1 / self._obj_radius
                                )
                                opposite_extreme_point = (
                                    obj_coord[0] + extreme_x_movement,
                                    obj_coord[1] + extreme_y_movement,
                                )

                                self._objects_coord[obj] = self.move_object(
                                    obj_coord, opposite_extreme_point, force
                                )

            # check that none of the objects are overlapping
            if any(
                (self.overlapping(new_held, o))
                for o in self._objects_coord.values()
                if o != new_held
            ):
                self._done = True

        return self.get_state(), self._done

    def init(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # get a type for each object based on the number of each type given in the constructor
        types = (
            [ObjectType.MAGNET] * self.n_magnets
            + [ObjectType.FERROMAGNETIC] * self.n_ferromagnetic
            + [ObjectType.NON_FERROMAGNETIC] * self.n_non_ferromagnetic
        )
        np.random.shuffle(types)  # randomize mapping of types to objects

        for i in range(self._total_objects):
            while True:
                x = np.random.rand()
                y = np.random.rand()
                x = max(0 + self._obj_radius, min(1 - self._obj_radius, x))
                y = max(0 + self._obj_radius, min(1 - self._obj_radius, y))
                if all(
                    self.distance((x, y), obj) > self._interaction_radius
                    for obj in self._objects_coord.values()
                ):
                    self._objects_coord[f"obj{i}"] = (x, y)
                    self._objects_coord_init[f"obj{i}"] = (x, y)
                    self._objects_type[f"obj{i}"] = types[i]
                    break

        return self.get_state()

    def reset(self):
        self._objects_coord = self._objects_coord_init.copy()
        self._done = False
        return self.get_state()

    def interacting(self, obj1_coord, obj2_coord, obj1_type, obj2_type):
        # The only cases where two objects interact are:
        if obj1_type == ObjectType.MAGNET and obj2_type == ObjectType.FERROMAGNETIC:
            types_interact = True
        elif obj1_type == ObjectType.FERROMAGNETIC and obj2_type == ObjectType.MAGNET:
            types_interact = True
        elif obj1_type == ObjectType.MAGNET and obj2_type == ObjectType.MAGNET:
            types_interact = True
        else:
            types_interact = False
        return (
            self.distance(obj1_coord, obj2_coord) <= self._interaction_radius
            and types_interact
        )

    def overlapping(self, obj1_coord, obj2_coord):
        # Consider the objects overlapping even if they touch
        return self.distance(obj1_coord, obj2_coord) <= 10e-6

    def distance(self, obj1_coord, obj2_coord):
        # compute the distance takin into account the radius
        x1, y1 = obj1_coord
        x2, y2 = obj2_coord
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) - self._obj_radius * 2

    def get_state(self):
        if len(self._objects_coord) == 0:
            raise ValueError(
                "The environment has not been initialized. Call init() method before getting the state."
            )
        else:
            return self._objects_coord.copy()

    def get_types(self):
        if len(self._objects_type) == 0:
            raise ValueError(
                "The environment has not been initialized. Call init() method before getting the state."
            )
        else:
            return self._objects_type.copy()

    def _generate_magnets_collection(self, objects, held_object=None):
        patches = []
        if len(objects) <= 10:
            cmap = plt.colormaps["tab10"]
            colors = cmap.colors
        else:
            cmap = plt.colormaps["viridis"]
            colors = cmap(
                np.linspace(0, 1, len(objects))
            )  # create a colorblind-friendly color map

        for i, obj in enumerate(objects):
            x, y = objects[obj]
            circle = Circle((x, y), self._obj_radius)
            patches.append(circle)

            if self._debug:
                # Get the object type
                obj_type = self._objects_type[obj]

                # map the object type to a string
                if obj_type == ObjectType.MAGNET:
                    obj_type = "M"
                elif obj_type == ObjectType.FERROMAGNETIC:
                    obj_type = "F"
                else:
                    obj_type = "N-F"

                # Add a text annotation for the object type
                text = obj_type
                if obj == held_object:
                    text += " (held)"
                plt.annotate(
                    text,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=24,
                )

        return PatchCollection(patches, color=colors)

    def render(self, held_object=None):
        return self.render_sequence([self._objects_coord], [held_object])

    def render_sequence(self, sequence, held_objects=None):
        assert len(sequence) == len(
            held_objects
        ), "The number of held objects must match the length of the sequence."

        subplot_width = 5  # width of each subplot in inches
        fig_height = 5  # height of the figure in inches

        # calculate the figure width based on the number of subplots
        fig_width = subplot_width * len(sequence)

        fig = plt.figure(figsize=(fig_width, fig_height))
        for i, objects in enumerate(sequence):
            ax = fig.add_subplot(1, len(sequence), i + 1)
            ax.add_collection(
                self._generate_magnets_collection(objects, held_objects[i])
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")

        # remove axis ticks
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        # create a canvas object and draw the figure on it
        canvas = FigureCanvas(fig)
        canvas.draw()

        # get the RGBA pixel data
        pixel_data = np.asarray(canvas.buffer_rgba())

        # convert the RGBA image to RGB
        pixel_data = np.delete(pixel_data, 3, axis=2)

        # close the figure to free up memory
        plt.close(fig)

        return pixel_data

    def print_image(self, img, size=5, show=True, path=None):
        self.print_image_sequence([img], size, show, path)

    def __iter__(self):
        if len(self._objects_coord) == 0:
            raise ValueError(
                "The environment has not been initialized. Call init() method before getting the state."
            )
        else:
            return iter(self._objects_coord)

    @staticmethod
    def print_image_sequence(img_list, size=5, show=True, path=None):
        n_images = len(img_list)
        fig = plt.figure(figsize=(size * n_images, size))
        for i, img in enumerate(img_list):
            ax = fig.add_subplot(1, n_images, i + 1)
            ax.imshow(img)
            ax.axis("off")

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        if path:
            plt.savefig(path, bbox_inches="tight", pad_inches=0)

        if show:
            plt.show()


def generate_sequence(
    n_magnets, n_ferromagnetic, n_non_ferromagnetic, n_steps, seed, debug=False
):
    env = MagnetsEnv(n_magnets, n_ferromagnetic, n_non_ferromagnetic, debug=debug)
    env.init(seed)
    coords = env.get_state()
    types = env.get_types()
    images = []
    held_objs = []
    step_count = 0
    done = False
    for obj in env:
        if step_count == 0:
            # Initialize the environment with the first object held
            images.append(env.render(obj))
            held_objs.append(obj)
            step_count += 1
        for target_obj in env:
            if done and step_count <= n_steps:
                env.reset()
                held_objs.append(obj)
                images.append(env.render(obj))
                step_count += 1
                done = False

            if obj != target_obj:
                while step_count <= n_steps and not done:
                    _, done = env.step(obj, target_obj)
                    held_objs.append(obj)
                    images.append(env.render(obj))
                    step_count += 1

    # fill the rest of the missing steps with the last image
    while step_count <= n_steps:
        images.append(env.render(held_objs[-1]))
        held_objs.append(held_objs[-1])
        step_count += 1

    assert len(images) == n_steps + 1 and len(images) == len(
        held_objs
    ), "Sequence generation failed!"

    # print the sequence of images
    if debug:
        env.print_image_sequence(images, show=True, path="magnets_sequence.png")

    return process(images, held_objs, types, coords)


def process(images, held, types, coords):
    new_images = []
    for image in images:
        new_image = image / 255.0
        new_image = E.reduce(
            new_image, "(h h2) (w w2) ... -> h w ...", "mean", h2=4, w2=4
        )

        new_images.append(new_image)
    images = tf.stack(new_images, axis=0)

    coords = [[x, y] for x, y in coords.values()]
    coords = tf.convert_to_tensor(coords, dtype=tf.float32)

    types = [v for v in types.values()]
    types = tf.convert_to_tensor(types, dtype=tf.int64)

    held = [int(h[-1]) for h in held[1:]]
    held = tf.convert_to_tensor(held, dtype=tf.int64)

    return images, held, types, coords


def generate_dataset(
    training_size,
    validation_size,
    test_size,
    n_magnets,
    n_ferromagnetic,
    n_non_ferromagnetic,
    n_steps,
    seed,
):
    dataset_dir = f"experiments/MAGNETS/data/m{n_magnets}_f{n_ferromagnetic}_nf{n_non_ferromagnetic}_steps{n_steps}_seed{seed}"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        os.chdir(dataset_dir)
        print_samples_num = 5

        samples_dir = "samples"
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)

        # generate training, validation, and test datasets
        print_samples_count = 0
        for i in range(training_size):
            print(f"Generating {i+1}/{training_size} training sample...", end="")
            training_sample = generate_sequence(
                n_magnets,
                n_ferromagnetic,
                n_non_ferromagnetic,
                n_steps,
                seed,
            )
            with open(
                f"training_set_m{n_magnets}_f{n_ferromagnetic}_nf{n_non_ferromagnetic}_steps{n_steps}_seq{i}.pkl",
                "wb",
            ) as f:
                pickle.dump(training_sample, f)

            if print_samples_count < print_samples_num:
                print_samples_count += 1
                info_path = f"{samples_dir}/training_set_m{n_magnets}_f{n_ferromagnetic}_nf{n_non_ferromagnetic}_steps{n_steps}_seq{i}"
                MagnetsEnv.print_image_sequence(
                    training_sample[0], show=False, path=f"{info_path}.png"
                )

                # print the coordinates of the objects and their types in a txt file
                with open(f"{info_path}.txt", "w") as f:
                    f.write(
                        f"Coordinates:\n{training_sample[1]}\n\nTypes:\n{training_sample[2]}"
                    )

            print("done!")
            seed += 1

        print_samples_count = 0
        for i in range(validation_size):
            print(f"Generating {i+1}/{validation_size} validation sample...", end="")
            validation_sample = generate_sequence(
                n_magnets,
                n_ferromagnetic,
                n_non_ferromagnetic,
                n_steps,
                seed,
            )
            with open(
                f"validation_set_m{n_magnets}_f{n_ferromagnetic}_nf{n_non_ferromagnetic}_steps{n_steps}_seq{i}.pkl",
                "wb",
            ) as f:
                pickle.dump(validation_sample, f)

            if print_samples_count < print_samples_num:
                print_samples_count += 1
                info_path = f"{samples_dir}/validation_set_m{n_magnets}_f{n_ferromagnetic}_nf{n_non_ferromagnetic}_steps{n_steps}_seq{i}"
                MagnetsEnv.print_image_sequence(
                    validation_sample[0], show=False, path=f"{info_path}.png"
                )

                # print the coordinates of the objects and their types in a txt file
                with open(f"{info_path}.txt", "w") as f:
                    f.write(
                        f"Coordinates:\n{validation_sample[1]}\n\nTypes:\n{validation_sample[2]}"
                    )

            print("done!")
            seed += 1

        print_samples_count = 0
        for i in range(test_size):
            print(f"Generating {i+1}/{test_size} test sample...", end="")
            test_sample = generate_sequence(
                n_magnets,
                n_ferromagnetic,
                n_non_ferromagnetic,
                n_steps,
                seed,
            )
            with open(
                f"test_set_m{n_magnets}_f{n_ferromagnetic}_nf{n_non_ferromagnetic}_steps{n_steps}_seq{i}.pkl",
                "wb",
            ) as f:
                pickle.dump(test_sample, f)

            if print_samples_count < print_samples_num:
                print_samples_count += 1
                info_path = f"{samples_dir}/test_set_m{n_magnets}_f{n_ferromagnetic}_nf{n_non_ferromagnetic}_steps{n_steps}_seq{i}"
                MagnetsEnv.print_image_sequence(
                    test_sample[0], show=False, path=f"{info_path}.png"
                )

                # print the coordinates of the objects and their types in a txt file
                with open(f"{info_path}.txt", "w") as f:
                    f.write(
                        f"Coordinates:\n{test_sample[1]}\n\nTypes:\n{test_sample[2]}"
                    )

            print("done!")
            seed += 1
    else:
        raise FileExistsError(f"Dataset {dataset_dir} already exists")


def make_tf_records(dataset_dir, tf_record_dir, model, split, size, horizon):
    print(f"Creating TFRecord files for the {split} set...")
    tf_record_file = tf_record_dir + f"/{split}.tfrecord"
    with tf.io.TFRecordWriter(tf_record_file) as writer:
        for i in range(size):
            with open(
                f"{dataset_dir}/{split}_set_m1_f1_nf1_steps{horizon}_seq{i}.pkl", "rb"
            ) as f:
                sequence = pickle.load(f)
                images, held, types = sequence
                if model == "deepsea":
                    image_0 = images[0]
                    next_images = images[1:]
                    probability = 1.0
                    features = image_0, types, next_images, held, probability

                    raise NotImplementedError("DeepSEA model not implemented yet")
                else:
                    feature = {
                        "images": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[images.numpy().tobytes()]
                            )
                        ),
                        "held": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[held.numpy().tobytes()]
                            )
                        ),
                        "types": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[types.numpy().tobytes()]
                            )
                        ),
                    }

                    example = tf.train.Example(
                        features=tf.train.Features(feature=feature)
                    )
                    writer.write(example.SerializeToString())

    print(f"TFRecord files for the {split} set created successfully!")


def sequence_to_dataset(sequence, model):
    images, held, types = sequence
    if model == "deepsea":
        image_0 = images[0]
        next_images = images[1:]
        probability = 1.0
        return image_0, types, next_images, held, probability
    elif model == "rnn":
        return images, held, types
    else:
        raise ValueError(f"Invalid model type {model}")


def get_generator_single(dataset_dir, split, horizon, index, model):
    with open(
        dataset_dir + "/" + f"{split}_set_m1_f1_nf1_steps{horizon}_seq{index}.pkl",
        "rb",
    ) as f:
        seq = pickle.load(f)
    return sequence_to_dataset(seq, model)


def get_generator(dataset_dir, split, horizon, size, model):
    for x in range(size):
        yield get_generator_single(dataset_dir, split, horizon, x, model)


# These are the latest generators that we actually use for speedup with TFRecords


def create_sample(
    n_steps,
    seed,
    n_magnets=1,
    n_ferromagnetic=2,
    n_non_ferromagnetic=0,
):
    sample = generate_sequence(
        n_magnets,
        n_ferromagnetic,
        n_non_ferromagnetic,
        n_steps,
        seed,
    )
    return sample


def print_sample(sample, tf_record_dir, split, index):
    info_path = f"{tf_record_dir}/{split}_sample_{index}"
    MagnetsEnv.print_image_sequence(sample[0], show=False, path=f"{info_path}.png")

    # print the coordinates of the objects and their types in a txt file
    with open(f"{info_path}.txt", "w") as f:
        f.write(f"\nHeld:\n{sample[1]}\n\nTypes:\n{sample[2]}")


def create_tf_records(tf_record_dir, split, size, horizon, seed_start):
    print(f"Creating TFRecord files for the {split} set...")
    tf_record_file = tf_record_dir + f"/{split}.tfrecord"
    with tf.io.TFRecordWriter(tf_record_file) as writer:
        for i in range(size):
            tf.print(f"Generating {i + 1}/{size} {split} sample...")
            images, held, types, coords = create_sample(horizon, seed=seed_start + i)

            if i < 5:
                print_sample((images, held, types), tf_record_dir, split, i)

            feature = {
                "images": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[images.numpy().tobytes()])
                ),
                "held": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[held.numpy().tobytes()])
                ),
                "types": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[types.numpy().tobytes()])
                ),
                "coords": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[coords.numpy().tobytes()])
                ),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    print(f"TFRecord files for the {split} set created successfully!")


def create_loader(
    horizon: int,
    batch_size: int,
    seed: int,
    training_size: int = 1000,
    validation_size: int = 100,
    test_size: int = 500,
    base_dir: str = "",
    equal_batches: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:

    # TODO: for code submission, remove name here!
    # tf_record_dir = f"nesy_mm/experiments/magnets/data/m1_f2_nf0_steps{horizon}_seed{seed}_tfrecord"
    tf_record_dir = f"/cw/dtailocal/lennert/deepseaproblog-dev/experiments/MAGNETS/data/m1_f2_nf0_steps{horizon - 1}_seed{seed}_tfrecord"

    if not os.path.exists(tf_record_dir):
        os.makedirs(tf_record_dir)

        create_tf_records(
            tf_record_dir, "training", training_size, horizon, seed_start=0
        )
        create_tf_records(
            tf_record_dir,
            "validation",
            validation_size,
            horizon,
            seed_start=training_size,
        )
        create_tf_records(
            tf_record_dir,
            "test",
            test_size,
            horizon,
            seed_start=training_size + validation_size,
        )

    feature_description = {
        "images": tf.io.FixedLenFeature([], tf.string),
        "held": tf.io.FixedLenFeature([], tf.string),
        "types": tf.io.FixedLenFeature([], tf.string),
        "coords": tf.io.FixedLenFeature([], tf.string),
    }

    if equal_batches:
        test_batch_size = batch_size
    else:
        test_batch_size = 50

    def _parse_function(example_proto):
        features = tf.io.parse_example(example_proto, feature_description)
        images = tf.io.decode_raw(features["images"], tf.float64)
        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, (batch_size, horizon, 125, 125, 3))
        held = tf.io.decode_raw(features["held"], tf.int64)
        held = tf.reshape(held, (batch_size, horizon - 1))
        types = tf.io.decode_raw(features["types"], tf.int64)
        types = tf.reshape(types, (batch_size, 3))
        types = tf.cast(types, tf.int32)
        coords = tf.io.decode_raw(features["coords"], tf.float32)
        coords = tf.reshape(coords, (batch_size, 3, 2))
        return images, held, types, coords

    def _parse_function_test(example_proto):
        features = tf.io.parse_example(example_proto, feature_description)
        images = tf.io.decode_raw(features["images"], tf.float64)
        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, (test_batch_size, horizon, 125, 125, 3))
        held = tf.io.decode_raw(features["held"], tf.int64)
        held = tf.reshape(held, (test_batch_size, horizon - 1))
        types = tf.io.decode_raw(features["types"], tf.int64)
        types = tf.reshape(types, (test_batch_size, 3))
        types = tf.cast(types, tf.int32)
        coords = tf.io.decode_raw(features["coords"], tf.float32)
        coords = tf.reshape(coords, (test_batch_size, 3, 2))
        return images, held, types, coords

    train_dataset = tf.data.TFRecordDataset(f"{tf_record_dir}/training.tfrecord")
    train_dataset = train_dataset.shuffle(500)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(
        _parse_function, num_parallel_calls=tf.data.AUTOTUNE
    )

    validation_dataset = tf.data.TFRecordDataset(f"{tf_record_dir}/validation.tfrecord")
    validation_dataset = validation_dataset.shuffle(100)
    validation_dataset = validation_dataset.batch(test_batch_size)
    validation_dataset = validation_dataset.map(
        _parse_function_test, num_parallel_calls=tf.data.AUTOTUNE
    )

    test_dataset = tf.data.TFRecordDataset(f"{tf_record_dir}/test.tfrecord")
    test_dataset = test_dataset.shuffle(200)
    test_dataset = test_dataset.batch(test_batch_size)
    test_dataset = test_dataset.map(
        _parse_function_test, num_parallel_calls=tf.data.AUTOTUNE
    )

    return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    # get CLI arguments with a parser and then call generate_dataset with the arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a dataset for the MAGNETS task"
    )
    parser.add_argument(
        "--training_size", type=int, help="Number of training samples to generate"
    )
    parser.add_argument(
        "--validation_size", type=int, help="Number of validation samples to generate"
    )
    parser.add_argument(
        "--test_size", type=int, help="Number of test samples to generate"
    )
    parser.add_argument(
        "--n_magnets", type=int, help="Number of magnets in the environment"
    )
    parser.add_argument(
        "--n_ferromagnetic",
        type=int,
        help="Number of ferromagnetic objects in the environment",
    )
    parser.add_argument(
        "--n_non_ferromagnetic",
        type=int,
        help="Number of non-ferromagnetic objects in the environment",
    )
    parser.add_argument("--n_steps", type=int, help="Number of steps in each sequence")
    parser.add_argument("--seed", type=int, help="Seed for the random number generator")

    args = parser.parse_args()

    generate_dataset(
        args.training_size,
        args.validation_size,
        args.test_size,
        args.n_magnets,
        args.n_ferromagnetic,
        args.n_non_ferromagnetic,
        args.n_steps,
        args.seed,
    )

    # To generate the image for the paper:
    # generate_sequence(1, 1, 1, 4, 12, debug=True)

    print("Dataset generated successfully")
