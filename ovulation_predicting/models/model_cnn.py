import numpy as np
import scipy.signal
import tensorflow as tf


class WideCnn(tf.keras.Model):
    def __init__(self, input_length, out_steps, num_features, min_peak_distance=20):
        """
        Model consisting of one convolutional layer with 256 filers and two dense layers (32, num_features).
        (Model makes one-step prediction and based on the desired output length feeds the prediction with the original
        input back to itself to generate the next output step.)
        Model has been simplified. Feedback loop is no longer present. Instead, a prediction of out_steps is made.
        Results are comparable to the original version and training is shorter.

        :param input_length: length of the input
        :param out_steps: output length
        :param num_features: number of input and output features
        :param min_peak_distance: minimum distance for peak detection
        """
        super().__init__()
        self.input_length = input_length
        self.out_steps = out_steps
        self.num_features = num_features
        self.num_output_features = num_features
        self.min_peak_distance = min_peak_distance
        conv_model_wide = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=256,
                                   kernel_size=input_length,
                                   activation='relu',
                                   input_shape=(input_length, num_features),),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(out_steps * num_features),
            tf.keras.layers.Reshape((out_steps, num_features))
        ])
        # lambda x: custom_activation(x, a=20.0)
        self.cnn = conv_model_wide

    def call(self, inputs):
        return self.cnn(inputs)

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
            "input_length": self.input_length,
            "out_steps": self.out_steps,
            "num_features": self.num_features,
            "min_peak_distance": self.min_peak_distance
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)