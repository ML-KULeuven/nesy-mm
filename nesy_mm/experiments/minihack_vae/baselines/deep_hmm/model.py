import tensorflow as tf

from nesy_mm.experiments.minihack_vae.baselines.deep_hmm.neural.transitions import NeuralTransition
from nesy_mm.experiments.minihack_vae.model import MinihackVAEModel


class MinihackVAEModelHMM(MinihackVAEModel):
    def __init__(self, grid_size, n_samples, latent_dim, image_shape, dropout, horizon, downsample=1, log_space=False):
        super().__init__(grid_size, n_samples, latent_dim, image_shape, dropout, horizon, downsample=downsample)
        self.transition = NeuralTransition(grid_size, log_space)

    def __call__(self, inputs, training=False, mask=None):
        return super().__call__(inputs, training, mask)
