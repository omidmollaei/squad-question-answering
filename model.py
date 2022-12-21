"""Implement the model using self and cross attention."""

import tensorflow as tf
from dataset import Parameters


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position: int, model_dim: int, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.model_dim = model_dim
        self.pos_encoding = self.positional_encoding(position, model_dim)

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({"position": self.position, "model_dim": self.model_dim})
        return config

    def get_angles(self, position: tf.Tensor, i: tf.Tensor, model_dim: tf.Tensor):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / model_dim)
        return position * angles

    def positional_encoding(self, position: int, model_dim: int):
        angle_rads = self.get_angles(
            position=tf.cast(tf.range(position)[:, tf.newaxis], dtype=tf.float32),
            i=tf.cast(tf.range(model_dim)[tf.newaxis, :], dtype=tf.float32),
            model_dim=tf.cast(model_dim, dtype=tf.float32))
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return pos_encoding

    def call(self, inputs: tf.Tensor):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]


def create_padding_mask(x: tf.Tensor):
    mask = tf.cast(tf.math.equal(x, 0), dtype=tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x: tf.Tensor):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0
    )
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)
