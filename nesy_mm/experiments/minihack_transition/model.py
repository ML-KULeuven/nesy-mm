import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import einops as E

from nesy_mm.experiments.minihack_vae.probabilistic.transitions import AgentTransition
from nesy_mm.experiments.minihack_transition.logic.transitions import MoveTransition
from nesy_mm.experiments.minihack_transition.logic.observations import Hit
from nesy_mm.experiments.minihack_transition.neural.transitions import (
    EnemyTransition,
    EnemyAction,
)

from nesy_mm.src.probabilistic.combinations import CombinationConstructor
from nesy_mm.src.probabilistic.resampling import FiniteResampler
from nesy_mm.src.probabilistic.disjoint_sum import disjoint_sum
from nesy_mm.src.utils import get_bernoulli_parameters


class LearnEnemyAction(tf.Module):
    def __init__(self, n_samples, log_space=False, relational=False):
        super().__init__()
        self.n_samples = n_samples
        self.log_space = log_space

        self.move_transition = MoveTransition()
        self.enemy_action = EnemyAction(log_space=log_space, relational=relational)
        self.hit = Hit()
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
        grid_size = inputs[5]

        # model hp of the agent
        agent_hp = [tf.ones([actions.shape[0], 1], dtype=tf.int32) * 12]

        enemies_location = tf.zeros([actions.shape[0], n_enemies, 2, grid_size])
        enemies_location = tfd.Categorical(logits=enemies_location)
        enemies_location = enemies_location.sample(self.n_samples)
        enemies_location = E.rearrange(enemies_location, "n b e d -> b e d n")

        agent_loc = agent_start_loc
        agent_loc = E.repeat(agent_loc, "b d -> b d n", n=self.n_samples)

        logits = 0.0
        for t in range(horizon - 1):
            # transition the agent
            logits_t = self.move_transition([agent_loc, actions[:, t], grid_size])

            agent_loc = tfd.Categorical(logits=logits_t)
            agent_loc = agent_loc.sample(1)[0]

            logits_t = tf.gather(logits_t, agent_loc, axis=-1, batch_dims=3)
            logits_t = tf.reduce_sum(logits_t, axis=-2)

            logits += logits_t

            # transition the enemies
            agent_loc = E.repeat(agent_loc, "b d n -> b e d n", e=n_enemies)

            action_logits_t = self.enemy_action(enemies_location, agent_loc)
            action_t = tfd.Categorical(logits=action_logits_t)
            action_t = action_t.sample(1)[0]

            enemy_logits = self.move_transition([enemies_location, action_t, grid_size])
            enemy_logits = tf.unstack(enemy_logits, axis=1)

            action_logits_t = tf.gather(
                action_logits_t, action_t, axis=-1, batch_dims=3
            )
            action_logits_t = tf.reduce_sum(action_logits_t, axis=-2)

            logits += action_logits_t

            # Potentially also parallelise the enemies here, might be the bottleneck
            # Also, probably want combinations of actions here, but only if memory runs out
            combinations, combinations_weights = self.combination_constructor(
                enemy_logits, [list(range(grid_size))] * n_enemies
            )

            agent_loc = E.rearrange(agent_loc, "b e d n -> b e 1 d n")

            combinations = tf.stack(combinations, axis=1)
            combinations = E.rearrange(combinations, "dim e combs -> 1 e combs dim 1")
            combination_observations = self.hit(agent_loc, combinations)
            combination_observations = combination_observations[..., 0]
            combination_observations = E.rearrange(
                combination_observations, "b e combs -> b combs e"
            )

            combinations = E.rearrange(combinations, "1 e combs dim 1 -> dim e combs")
            combinations = tf.unstack(combinations, axis=1)

            # TODO: check if disjoint sum impacts performance
            combination_observations = disjoint_sum(combination_observations, axis=-1)
            combination_observations = tf.expand_dims(combination_observations, axis=-2)
            combination_observations = get_bernoulli_parameters(
                combination_observations, log_space=True
            )

            observation = enemy_hit[:, t + 1]
            observation = E.repeat(
                observation,
                "b -> b 1 combinations",
                combinations=combination_observations.shape[-2],
            )

            enemies_location, exact_conditionals = self.resampler(
                "categorical",
                combination_observations,
                observation,
                combinations_weights,
                combinations,
            )
            enemies_location = tf.stack(enemies_location, axis=1)

            exact_conditionals = tf.stack(exact_conditionals, axis=1)
            exact_conditionals = tf.reduce_sum(exact_conditionals, axis=1)

            logits += exact_conditionals

            agent_loc = agent_loc[:, :, 0]
            mask, hit_logits = self.count_hits(agent_loc, enemies_location)
            agent_loc = agent_loc[:, 0]

            logits += hit_logits

            # We use an imp, which is one d4 damage
            damage = self.get_damage(mask, n_enemies, 1, 4)
            agent_hp.append(agent_hp[-1] - damage)

        dead_t = tf.cast(agent_hp[-1] <= 0, tf.int32)
        return dead_t, logits

    def save_weights(self, path):
        self.enemy_action.save_weights(f"{path}.enemy_action")
        self.hit.save_weights(f"{path}.hit")

    def load_weights(self, path):
        self.enemy_action.load_weights(f"{path}.enemy_action")
        self.hit.load_weights(f"{path}.hit")
