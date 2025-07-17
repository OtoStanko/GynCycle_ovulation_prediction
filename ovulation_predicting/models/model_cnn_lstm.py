import numpy as np
import scipy.signal
import tensorflow as tf


class CnnLstm(tf.keras.Model):
    def __init__(self, units, input_length, out_steps, num_features, min_peak_distance=20,
                 filters=None, ks=None, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4]
        if ks is None:
            ks = [4, 3, 2]
        if filters is None:
            filters = [32, 64, 128]
        self.units = units
        self.input_length = input_length
        self.out_steps = out_steps
        self.num_features = num_features
        self.num_output_features = num_features
        self.min_peak_distance = min_peak_distance
        self.cnl = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=filters[0], kernel_size=ks[0], activation='relu', padding='same',
                                   dilation_rate=dilations[0],
                                   input_shape=(input_length, num_features)),
            tf.keras.layers.Conv1D(filters=filters[1], kernel_size=ks[1], activation='relu', padding='same',
                                   dilation_rate=dilations[1]),
            tf.keras.layers.Conv1D(filters=filters[2], kernel_size=ks[2], activation='relu', padding='same',
                                   dilation_rate=dilations[2]),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(out_steps * num_features),
            tf.keras.layers.Reshape((out_steps, num_features))
        ])

    def call(self, inputs):
        return self.cnl(inputs)

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
