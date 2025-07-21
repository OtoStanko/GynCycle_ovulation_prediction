import numpy as np
import scipy.signal
from scipy.signal import savgol_filter
import tensorflow as tf


class ClassificationMLP(tf.keras.Model):
    def __init__(self, input_length, out_steps, num_features, min_peak_distance):
        """
        Classification model with 3 dense layers (256, 64, out_steps). Only outputs one feature where peaks should
        be detected. Can take in multiple features inputs. The output layer has sigmoid activation function. The output
        can thus be interpreted as probability of the peak being at that position.

        :param input_length: length of the input
        :param out_steps: output length
        :param num_features: number of input features - num_output_features = 1
        :param min_peak_distance: minimum distance for peak detection
        """
        super().__init__()
        self.input_length = input_length
        self.out_steps = out_steps
        self.num_features = num_features
        self.num_output_features = 1
        self.min_peak_distance = min_peak_distance
        self.mlp = tf.keras.models.Sequential([
            tf.keras.layers.Reshape((1, num_features * input_length)),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=out_steps, activation='sigmoid'),
        ])

    def call(self, inputs):
        inputs = tf.reshape(inputs, (-1, self.num_features, self.input_length))
        shape = inputs.shape
        print(shape)
        result = self.mlp(inputs)
        print(result.shape)
        return result

    def get_peaks(self, prediction, method='raw'):
        """
        For given model predictions identifies peaks in it. Allows multiple methods for peak detection in the prediction.


        :param prediction: ndarray of predictions (one feature)
        :param method: 'raw' - treats the prediction as it would be hormone levels. Finds peaks based on the height
        and distance from other peaks. 'smooth' - applies smoothing to the output and then finds peaks as previously
        'combined' - finds peaks as in raw method. Keeps only the first one. In the rest of the prediction finds the highest
        value that can be peak (left and right values are lower) and adds it into the results. Always returns at most two peaks.
        :return: ndarray of indexes where peaks were detected in the input array
        """
        if method == 'raw':
            return self.peaks_raw(prediction, self.min_peak_distance)
        elif method == 'smooth':
            return self.peaks_smoothened(prediction, self.min_peak_distance)
        elif method == 'combined':
            return self.peaks_combined(prediction, self.min_peak_distance)

    def peaks_raw(self, prediction, min_peak_distance):
        pred_peaks, _ = scipy.signal.find_peaks(prediction, distance=min_peak_distance)
        position_of_max = np.argmax(prediction)
        # The following code sometimes results in 2 peaks too close to each other, but that is somehow acceptable
        if position_of_max not in pred_peaks:
            index = np.searchsorted(pred_peaks, position_of_max)
            pred_peaks = np.insert(pred_peaks, index, position_of_max)
        return pred_peaks

    def peaks_smoothened(self, prediction, min_peak_distance):
        result = tf.reshape(prediction, (self.out_steps))
        result = result / 3
        result = savgol_filter(result, 11, 2)
        pred_peaks, _ = scipy.signal.find_peaks(result, distance=min_peak_distance)
        return pred_peaks

    def is_peak(self, index, values):
        if index == 0:  # First element
            return values[index] > values[index + 1]
        elif index == len(values) - 1:  # Last element
            return values[index] > values[index - 1]
        else:  # Middle elements
            return values[index] > values[index - 1] and values[index] > values[index + 1]

    def peaks_combined(self, prediction, min_peak_distance):
        offset = 15
        pred_peaks, _ = scipy.signal.find_peaks(prediction, distance=min_peak_distance)
        first_peak = pred_peaks[0]
        result = np.array([first_peak])
        if first_peak + offset < 35:
            potential_second_peak = prediction[first_peak+offset:]
            sorted_indexes = np.argsort(potential_second_peak)[::-1]
            sorted_indexes += first_peak + offset
            peak_index = None
            for index in sorted_indexes:
                if self.is_peak(index, prediction):
                    peak_index = index
                    break
            if peak_index is not None:
                result = np.append(result, peak_index)
        return result

    def get_config(self):
        # Return the configuration of the model (needed for saving and loading)
        config = super().get_config().copy()
        config.update({
            "input_length": self.input_length,
            "out_steps": self.out_steps,
            "num_features": self.num_features,
            "min_peak_distance": self.min_peak_distance
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)