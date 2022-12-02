"""Implement the model using self attention , cross attention and bidirectional gru."""

import tensorflow as tf
from tensorflow import keras
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
            model_dim=tf.cast(model_dim, dtype=tf.float32),
        )

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return pos_encoding

    def call(self, inputs: tf.Tensor):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]


def self_atten_block(params: Parameters, name: str = "self_attention_block"):
    inputs = keras.layers.Input(shape=[None, params.model_dim], name="inputs")
    atten_mask = keras.layers.Input(shape=[None, None], name="attention_mask")
    atten = keras.layers.MultiHeadAttention(
        key_dim=params.model_dim,
        num_heads=params.num_heads,
    )(
        query=inputs,
        value=inputs,
        attention_mask=atten_mask
    )

    atten = keras.layers.Dropout(rate=params.dropout_rate)(atten)
    atten = keras.layers.Add()([atten, tf.cast(inputs, dtype=tf.float32)])
    atten = keras.layers.LayerNormalization(epsilon=1e-6)(atten)

    outputs = keras.layers.Dense(params.dense_units, activation=params.activation)(atten)
    outputs = keras.layers.Dense(params.model_dim)(outputs)
    outputs = keras.layers.Dropout(rate=params.dropout_rate)(outputs)
    outputs = keras.layers.Add()([atten, tf.cast(inputs, dtype=tf.float32)])
    outputs = keras.layers.LayerNormalization(epsilon=1e-6)(outputs)

    return tf.keras.Model(inputs=[inputs, atten_mask], outputs=outputs)


def self_atten_encoder(params: Parameters, name: str = "self_atten_encoder"):
    inputs = keras.layers.Input(shape=[None, ], name="inputs")
    atten_mask = keras.layers.Input(shape=[None, None], name="atten_mask")
    embeddings = keras.layers.Embedding(params.vocab_size, params.model_dim, mask_zero=True)
    embeddings *= tf.math.sqrt(tf.cast(params.model_dim, dtype=tf.float32))
    embeddings = PositionalEncoding(
        position=params.vocab_size, model_dim=params.model_dim
    )(embeddings)
    outputs = keras.layers.Dropout(rate=params.dropout_rate)(embeddings)

    for i in range(params.atten_encoder_num_layers):
        outputs = self_atten_block(
            params=params,
            name=f"self_atten_block_{i}"
        )([outputs, atten_mask])

    return tf.keras.Model(inputs=[inputs, atten_mask], outputs=outputs)


def recurrent_block(params: Parameters, name: str = "recurrent_encoder_block"):
    inputs = keras.layers.Input(shape=[None, params.model_dim])
    outputs = keras.layers.Bidirectional(
        merge_mode="sum",
        layer=keras.layers.GRU(
            units=params.model_dim, return_sequences=True, recurrent_initializer="glorot_uniform"
        )
    )(inputs)

    return tf.keras.Model(inputs, outputs, name=name)


def recurrent_encoder(params: Parameters, name: str = "recurrent_encoder"):
    inputs = keras.layers.Input(shape=[None, ], name="inputs")
    embeddings = keras.layers.Embedding(params.vocab_size, params.model_dim)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(params.model_dim, dtype=tf.float32))
    outputs = PositionalEncoding(
        position=params.vocab_size, d_model=params.model_dim
    )(embeddings)

    for i in range(params.recurrent_encoder_num_layers):
        outputs = recurrent_block(
            params=params,
            name=f"recurrent_encoder_block_{i}"
        )(outputs)

    return tf.keras.Model(inputs, outputs)
