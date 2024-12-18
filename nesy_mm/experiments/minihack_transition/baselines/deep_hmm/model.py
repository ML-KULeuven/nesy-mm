import einops as E
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd

from nesy_mm.src.probabilistic.resampling import FiniteResampler
from nesy_mm.src.utils import get_bernoulli_parameters, add_batch_dimension_like
from nesy_mm.src.probabilistic.combinations import CombinationConstructor
from nesy_mm.src.probabilistic.disjoint_sum import disjoint_sum

from nesy_mm.experiments.minihack_transition.neural.transitions import EnemyTransition
from nesy_mm.experiments.minihack_transition.baselines.deep_hmm.neural.transitions import (
    NeuralTransition,
)
from nesy_mm.experiments.minihack_transition.baselines.deep_hmm.neural.observations import (
    NeuralHit,
)


class LearnEnemyTransitionHMM(tf.Module):
    def __init__(self, grid_size, n_samples, log_space=False):
        super().__init__()
        self.grid_size = grid_size + 2
        self.n_samples = n_samples
        self.log_space = log_space

        self.agent_transition = NeuralTransition(grid_size, log_space)
        self.enemy_transition = EnemyTransition(grid_size, log_space=log_space)
        self.hit = NeuralHit(grid_size)
        self.combination_constructor = CombinationConstructor()
        self.resampler = FiniteResampler()

    def get_damage(self, mask, n_enemies, n_dice, die_value):
        damage = tf.zeros([1, n_enemies, die_value])
        damage = tfd.Categorical(logits=damage).sample(self.n_samples * n_dice) + 1
        damage = E.rearrange(damage, "(dice n) b e -> b e dice n", n=self.n_samples)
        damage = tf.reduce_sum(damage, axis=-2)
        damage = damage * mask
        damage = tf.reduce_sum(damage, axis=-2)
        return damage

    def count_hits(self, agent_loc, enemies_location):
        enemy_hit_logits = self.hit(agent_loc, enemies_location)
        enemy_hit_logits = get_bernoulli_parameters(enemy_hit_logits, log_space=True)

        enemy_hit = tfd.Categorical(logits=enemy_hit_logits)
        enemy_hit = enemy_hit.sample(1)[0]

        enemy_hit_logits = tf.gather(enemy_hit_logits, enemy_hit, axis=-1, batch_dims=3)
        enemy_hit_logits = tf.reduce_sum(enemy_hit_logits, axis=-2)

        return enemy_hit, enemy_hit_logits

    def __call__(self, inputs, training=False, mask=None):
        actions = inputs[0]  # shape: (batch, horizon - 1, 4)
        agent_start_loc = inputs[1]  # shape: (batch, 2)
        enemy_hit = inputs[2]  # shape: (batch, horizon, 1)

        horizon = inputs[3]
        n_enemies = inputs[4]

        # model hp of the agent
        agent_hp = [tf.ones([actions.shape[0], 1], dtype=tf.int32) * 12]

        # model location and hp of the enemies
        enemies_location = tf.zeros([actions.shape[0], n_enemies, 2, self.grid_size])
        enemies_location = tfd.Categorical(logits=enemies_location)
        enemies_location = enemies_location.sample(self.n_samples)
        enemies_location = E.rearrange(enemies_location, "n b e d -> b e d n")

        agent_loc = agent_start_loc
        agent_loc = E.repeat(agent_loc, "b d -> b d n", n=self.n_samples)

        logits = 0
        for t in range(horizon - 1):
            # transition the agent
            agent_logits_t = self.agent_transition([agent_loc, actions[:, t]])

            # transition the enemies
            agent_loc = E.repeat(agent_loc, "b d n -> b e d n", e=n_enemies)

            enemy_logits = self.enemy_transition(enemies_location, agent_loc)
            enemy_logits = tf.unstack(enemy_logits, axis=1)

            combination_logits = enemy_logits + [agent_logits_t]
            combinations, combinations_weights = self.combination_constructor(
                combination_logits,
                [list(range(self.grid_size))] * (n_enemies + 1),
            )

            agent_loc = combinations[-1]
            agent_loc = E.rearrange(
                agent_loc, "dim combinations -> combinations 1 dim 1"
            )

            enemy_combinations = tf.stack(combinations[:-1], axis=1)
            enemy_combinations = E.rearrange(
                enemy_combinations, "dim e combs -> combs e dim 1"
            )

            combination_observations = self.hit(agent_loc, enemy_combinations)
            combination_observations = combination_observations[..., 0]

            combination_observations = disjoint_sum(combination_observations, axis=-1)
            combination_observations = tf.expand_dims(combination_observations, axis=-2)
            combination_observations = get_bernoulli_parameters(
                combination_observations, log_space=True
            )
            combination_observations = add_batch_dimension_like(
                combination_observations, actions[:, 0, 0]
            )

            observation = enemy_hit[:, t + 1]
            observation = E.repeat(
                observation,
                "b -> b 1 combinations",
                combinations=combination_observations.shape[-2],
            )

            entity_location, exact_conditionals = self.resampler(
                "categorical",
                combination_observations,
                observation,
                combinations_weights,
                combinations,
            )
            enemies_location = entity_location[:-1]
            enemies_location = tf.stack(enemies_location, axis=1)

            agent_loc = entity_location[-1]
            agent_loc = E.repeat(agent_loc, "b d n -> b e d n", e=n_enemies)

            exact_conditionals = tf.stack(exact_conditionals, axis=1)
            exact_conditionals = tf.reduce_sum(exact_conditionals, axis=1)

            logits += exact_conditionals

            mask, hit_logits = self.count_hits(agent_loc, enemies_location)
            agent_loc = agent_loc[:, 0]

            logits += hit_logits

            # We use an imp, which is one d4 damage (probably should be replaced somehow with learnable parameters)
            damage = self.get_damage(mask, n_enemies, 1, 4)
            agent_hp.append(agent_hp[-1] - damage)

            del combinations
            del combination_observations
            del combinations_weights

        dead_t = tf.cast(agent_hp[-1] <= 0, tf.int32)
        return dead_t, logits

    def save_weights(self, path):
        self.agent_transition.save_weights(f"{path}.agent_transition")
        self.enemy_transition.save_weights(f"{path}.enemy_transition")
        self.hit.save_weights(f"{path}.hit")

    def load_weights(self, path):
        self.agent_transition.load_weights(f"{path}.agent_transition")
        self.enemy_transition.load_weights(f"{path}.enemy_transition")
        self.hit.load_weights(f"{path}.hit")
