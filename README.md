# Relational Neurosymbolic Markov Models (NeSy-MMs)

This repository contains the code for the paper _Relational Neurosymbolic Markov Models_ [[arXiv](https://arxiv.org/abs/2412.13023)].

## Installation

To install the required packages, we advise using `uv`, the modern, Rust-based Python package manager.
To install `uv`, use

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Next, use `uv` to create a virtual environment and install the dependencies with

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

Notice that the commands provided here are going to generate the dataset and then train the model, so it may take a while the first time.
To generate the dataset in a different directory you can use the option `--base_dir <path>`.
To use the [Weights & Biases](http://wandb.ai/) logging, you can use the option `--wandb_mode "online"`.
You can specify the seed with the option `--seed <seed>`. If you don't the default seed (42) is used.

### Generative Experiment

To run the generative experiment, use the following commands, depending on what model you want to train.
Training on an NVIDIA Tesla P100 GPU should take between 4 and 10 hours (per run, per model) depending on the model.

```bash
python nesy_mm/experiments/minihack_vae/run.py --grid_size 5
python nesy_mm/experiments/minihack_vae/run.py --grid_size 10 --downsample 2 --n_samples 20 --batch_size 5 --beta 50 --n_epochs 15

python nesy_mm/experiments/minihack_vae/baselines/deep_hmm/run.py --grid_size 5
python nesy_mm/experiments/minihack_vae/baselines/deep_hmm/run.py --grid_size 10 --downsample 2 --n_samples 20 --batch_size 5 --n_epochs 15

python nesy_mm/experiments/minihack_vae/baselines/transformer/run.py --grid_size 5
python nesy_mm/experiments/minihack_vae/baselines/transformer/run.py --grid_size 10 --downsample 2
```

To test the generative models, there is a Jupiter notebook `generation.ipynb` that you can use. This can be easily run on CPU.

### Transition Learning Experiment

To run the transition learning experiment, use the following commands, depending on what model you want to train.

```bash
python nesy_mm/experiments/minihack_transition/run.py
python nesy_mm/experiments/minihack_transition/evaluate.py

python nesy_mm/experiments/minihack_transition/baselines/deep_hmm/run.py
python nesy_mm/experiments/minihack_transition/baselines/deep_hmm/evaluate.py

python nesy_mm/experiments/minihack_transition/baselines/transformer/run.py
python nesy_mm/experiments/minihack_transition/baselines/transformer/evaluate.py
```

In this case evaluation of the models is performed separately from the training script because we evaluate on multiple
out-of-distribution datasets.
