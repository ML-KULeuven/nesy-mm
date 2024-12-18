import os
import tensorflow as tf
import einops as E
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import *


class BaseAttention(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):

    def __call__(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x, key=context, value=context, return_attention_scores=True
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):

    def __call__(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):

    def __call__(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class Classifier(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential()
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))

    def __call__(self, x, training=False, mask=None):
        x = self.model(x)
        return x


class Mirage(tf.Module):

    def __init__(self, grid_size, embedding_size, num_heads=8, key_dim=64, dropout=0.1):
        super().__init__()
        self.grid_size = grid_size + 2
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout

        self.dropout = Dropout(dropout)

        self.causal_attn = CausalSelfAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout
        )

        self.cross_attn = CrossAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout
        )

        self.classifier = Classifier()

    def next(self, input_sequence, context):
        x = self.dropout(input_sequence)
        x = self.causal_attn(x)
        x = self.cross_attn(x, context)
        return x[:, -1, ...]

    def __call__(self, inputs, training=False, mask=None):
        actions = inputs[0]
        actions = tf.cast(actions, tf.keras.backend.floatx())
        dummy_action = tf.ones_like(actions[:, 0:1, ...]) * -1.0
        actions = tf.concat([dummy_action, actions], axis=1)

        agent_start_loc = inputs[1]
        agent_start_loc = tf.cast(agent_start_loc, tf.keras.backend.floatx())

        enemy_hit = inputs[2]
        enemy_hit = tf.cast(enemy_hit, tf.keras.backend.floatx())
        enemy_hit = tf.expand_dims(enemy_hit, axis=-1)

        start_embedding = np.zeros([agent_start_loc.shape[0], self.embedding_size - 2])
        start_embedding = tf.concat([agent_start_loc, start_embedding], axis=-1)

        context = tf.concat([actions, enemy_hit], axis=-1)

        horizon = actions.shape[1]

        output_array = tf.TensorArray(dtype=tf.keras.backend.floatx(), size=horizon + 1)
        output_array = output_array.write(0, start_embedding)

        for i in tf.range(horizon):
            output = output_array.stack()
            output = E.rearrange(output, "t b e -> b t e")
            next = self.next(output, context)
            output_array = output_array.write(i + 1, next)

        output = output_array.stack()
        # output = E.rearrange(output, "t b e -> b (t e)")
        output = output[-1, ...]
        output = self.classifier(output)
        return output

    def load_weights(self, path):
        self.classifier.load_weights(f"{path}.classifier")
        self.causal_attn.load_weights(f"{path}.causal_attn")
        self.cross_attn.load_weights(f"{path}.cross_attn")

    def save_weights(self, path):
        self.classifier.save_weights(f"{path}.classifier")
        self.causal_attn.save_weights(f"{path}.causal_attn")
        self.cross_attn.save_weights(f"{path}.cross_attn")
