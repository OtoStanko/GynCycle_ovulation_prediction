import numpy as np
import scipy.signal
import tensorflow as tf


class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_features, min_peak_distance=20):
        """
        A feedback model consisting of one RNN layer with LSTM cell and one dense layer.

        :param units: number of units in the LSTM cell
        :param out_steps: output length
        :param num_features: number of input and output features
        :param min_peak_distance: minimum distance for peak detection
        """
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.num_features = num_features
        self.num_output_features = num_features
        self.min_peak_distance = min_peak_distance
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        predictions = []
        prediction, state = self.warmup(inputs)
        predictions.append(prediction)

        for n in range(1, self.out_steps):
            x = prediction
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            prediction = self.dense(x)
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

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

    def get_config(self):
        # Return the configuration of the model (needed for saving and loading)
        config = super().get_config().copy()
        config.update({
            "units": self.units,
            "out_steps": self.out_steps,
            "num_features": self.num_features,
            "min_peak_distance": self.min_peak_distance
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)