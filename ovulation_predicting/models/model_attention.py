import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow.keras import layers


class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_len, d_model):
        super().__init__()
        self.pos_encoding = self.get_positional_encoding(sequence_len, d_model)

    def get_positional_encoding(self, seq_len, d_model):
        angle_rads = self._get_angles(np.arange(seq_len)[:, np.newaxis],
                                      np.arange(d_model)[np.newaxis, :],
                                      d_model)
        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class Attention(tf.keras.Model):
    def __init__(self, units, input_length, out_steps, num_features, min_peak_distance=20):
        super().__init__()

        self.units = units
        self.input_length = input_length
        self.out_steps = out_steps
        self.num_features = num_features
        self.num_output_features = num_features
        self.min_peak_distance = min_peak_distance

        self.embedding = layers.Dense(units)
        self.pos_encoding = PositionalEncoding(input_length, units)
        self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=units)
        self.dropout1 = layers.Dropout(0.1)
        self.norm1 = layers.LayerNormalization()

        self.ffn = tf.keras.Sequential([
            layers.Dense(2*units, activation='relu'),
            layers.Dense(units),
        ])
        self.dropout2 = layers.Dropout(0.1)
        self.norm2 = layers.LayerNormalization()

        self.output_dense = layers.Dense(out_steps * num_features)
        self.reshape = layers.Reshape((out_steps, num_features))

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.pos_encoding(x)

        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output, training=training))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output, training=training))
        x = tf.reduce_mean(x, axis=1)

        x = self.output_dense(x)
        x = self.reshape(x)
        return x

    def get_peaks(self, prediction, method='raw'):
        """
        For given model predictions identifies peaks in it.

        :param prediction: ndarray of predictions (one feature)
        :param method: NA for this model
        :return: ndarray of indexes where peaks were detected in the input array
        """
        pred_peaks, _ = scipy.signal.find_peaks(prediction, distance=self.min_peak_distance, height=0.2)
        position_of_max = np.argmax(prediction)
        if position_of_max not in pred_peaks:
            index = np.searchsorted(pred_peaks, position_of_max)
            pred_peaks = np.insert(pred_peaks, index, position_of_max)
        return pred_peaks
