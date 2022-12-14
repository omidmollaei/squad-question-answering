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


def scaled_dot_product_attention(query, key, value, mask):
    """Perform attention mechanism and calculate attention weights"""
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], dtype=tf.float32)  # scale matmul_qk
    logits = matmul_qk / tf.math.sqrt(depth)
    if mask is not None:   # add the mask to zero out padding tokens
        logits += mask * -1e9
    attention_weights = tf.nn.softmax(logits, axis=-1)  # softmax is normalized on the last axis (seq_len_k)
    output = tf.matmul(attention_weights, value)
    return output


class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads: int, model_dim: int, **kwargs):
        assert model_dim % num_heads == 0
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.depth = self.model_dim // self.num_heads
        self.query_dense = tf.keras.layers.Dense(self.model_dim)
        self.key_dense = tf.keras.layers.Dense(self.model_dim)
        self.value_dense = tf.keras.layers.Dense(self.model_dim)
        self.dense = tf.keras.layers.Dense(self.model_dim)

    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({"num_heads": self.num_heads, "model_dim": self.model_dim})
        return config

    def split_head(self, inputs: tf.Tensor, batch_size: int):
        inputs = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, shape=(batch_size, -1, self.num_heads, self.depth)))(inputs)
        return tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(inputs)

    def call(self, inputs: tf.Tensor):
        query, key, value, mask = (inputs['query'], inputs['key'], inputs['value'], inputs['mask'])
        batch_size = tf.shape(query)[0]

        # linear layer
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_head(query, batch_size)
        key = self.split_head(key, batch_size)
        value = self.split_head(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(scaled_attention)

        concat_attention = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, (batch_size, -1, self.model_dim)))(scaled_attention)  # concatenation of heads

        outputs = self.dense(concat_attention)  # final linear layer
        return outputs


def encoder_layer(params: Parameters, name: str = "encoder_layer"):
    inputs = tf.keras.layers.Input(shape=(None, params.model_dim), name="inputs")
    padding_mask = tf.keras.layers.Input(shape=(1, 1, None), name="padding_mask")
    attention = MultiHeadAttentionLayer(
        num_heads=params.num_heads, model_dim=params.model_dim, name="self_attention"
    )({
        "query": inputs,
        "key": inputs,
        "value": inputs,
        "mask": padding_mask
    })
    attention = tf.keras.layers.Dropout(params.dropout_rate)(attention)
    attention += tf.cast(inputs, dtype=tf.float32)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention)
    outputs = tf.keras.layers.Dense(params.num_units, activation=params.activation)(attention)
    outputs = tf.keras.layers.Dense(params.model_dim)(outputs)
    outputs = tf.keras.layers.Dropout(params.dropout_rate)(outputs)
    outputs += attention
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs)
    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(params: Parameters, name: str = "encoder"):
    inputs = tf.keras.layers.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.layers.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(params.vocab_size, params.model_dim)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(params.model_dim, dtype=tf.float32))
    embeddings = PositionalEncoding(position=params.vocab_size, model_dim=params.model_dim)(embeddings)

    outputs = tf.keras.layers.Dropout(params.dropout_rate)(embeddings)
    for i in range(params.num_layers):
        outputs = encoder_layer(params, name=f"{name}_layer_no_{i}")([outputs, padding_mask])
    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(params: Parameters, name: str = "decoder_layer"):
    inputs = tf.keras.layers.Input(shape=(None, params.model_dim), name="inputs")
    enc_outputs = tf.keras.layers.Input(shape=(None, params.model_dim), name="encoder_outputs")
    look_ahead_mask = tf.keras.layers.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.layers.Input(shape=(1, 1, None), name="padding_mask")

    attention1 = MultiHeadAttentionLayer(
        num_heads=params.num_heads, model_dim=params.model_dim, name="attention_1"
    )(inputs={"query": inputs, "key": inputs, "value": inputs, "mask": look_ahead_mask})
    attention1 += tf.cast(inputs, dtype=tf.float32)
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1)

    attention2 = MultiHeadAttentionLayer(
        num_heads=params.num_heads, model_dim=params.model_dim, name="attention_2"
    )(inputs={"query": attention1, "key": enc_outputs, "value": enc_outputs, "mask": padding_mask})
    attention2 = tf.keras.layers.Dropout(params.dropout_rate)(attention2)
    attention2 += attention1
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2)

    outputs = tf.keras.layers.Dense(params.num_units, activation=params.activation)(attention2)
    outputs = tf.keras.layers.Dense(params.model_dim)(outputs)
    outputs = tf.keras.layers.Dropout(params.dropout_rate)(outputs)
    outputs += attention2
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs)

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
                          outputs=outputs, name=name)


def decoder(params: Parameters, name: str = "decoder"):
    inputs = tf.keras.layers.Input(shape=(None,), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, params.model_dim), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(params.vocab_size, params.model_dim)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(params.model_dim, dtype=tf.float32))
    embeddings = PositionalEncoding(position=params.vocab_size, model_dim=params.model_dim)(embeddings)

    outputs = tf.keras.layers.Dropout(params.dropout_rate)(embeddings)
    for i in range(params.num_layers):
        outputs = decoder_layer(
            params, name=f"{name}_layer_no_{i}"
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
                          outputs=outputs, name=name)


def transformer(params: Parameters, name: str = "transformer"):
    inputs = tf.keras.layers.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.layers.Input(shape=(None,), name="dec_inputs")

    enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),
                                              name="enc_padding_mask")(inputs)
    look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None),
                                             name="look_ahead_mask")(dec_inputs)
    dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),
                                              name="dec_padding_mask")(inputs)

    enc_outputs = encoder(params)(inputs=[inputs, enc_padding_mask])
    dec_outputs = decoder(params)(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(params.vocab_size, name="outputs")(dec_outputs)
    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, model_dim: int, warmup_steps: int = 4000):
        super(LearningRateSchedule, self).__init__()
        self.model_dim = tf.cast(model_dim, dtype=tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * self.warmup_steps**-1.5
        return tf.math.rsqrt(self.model_dim) * tf.math.minimum(arg1, arg2)
