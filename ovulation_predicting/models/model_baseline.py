from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.signal
import tensorflow as tf

from ovulation_predicting.supporting_scripts import sin_function


class NoisySinCurve(tf.keras.Model):
    def __init__(self, input_length, out_steps, num_features, train_df, feature,
                 noise=0, shift=0, period=28, min_peak_distance=20):
        """
        Sine curve meant as a baseline for LH prediction. The model cannot be trained. When instantiated a train_df
        with one feature must be provided. This feature will be used to set the function period (mean peak distance).

        Allows adding noise into the output from normal distribution

        :param input_length: length of the input
        :param out_steps: output length
        :param num_features: number of input and output features (should be 1)
        :param train_df: dataset used to determining the period length (in days)
        :param feature: feature from the train_df where peaks should be predicted
        :param noise: scale (std) of normal distribution with mean=0
        :param shift: initial shift along the x-axis
        :param period: initial period (in days)
        :param min_peak_distance: minimum distance for peak detection
        """
        super().__init__()
        self.input_length = input_length
        self.out_steps = out_steps
        self.num_features = num_features
        self.num_output_features = 1
        self.noise = noise / 10
        self.period = period
        self.shift = shift
        self.min_peak_distance = min_peak_distance
        x_data = train_df.index.values
        y_data = train_df[feature].values
        popt, _ = curve_fit(self.move_curve_function, x_data, y_data, p0=[self.shift])
        self.shift = popt
        print(f"Optimal parameters: b={self.shift}, c={self.period}")
        print('0.05 * sin( (x-b) * ((2*pi)/(c*24)) ) + 0.05')
        x_fit = np.linspace(1200, 3500, 100)
        y_fit = sin_function(x_fit, self.shift, self.period)
        plt.plot(train_df.index[:100], train_df[feature][:100], color='black')
        plt.plot(x_fit, y_fit, label='Fitted Curve', color='orange')
        plt.title('Sampled dataframe with raw hours with fitted sin curve')
        plt.xlabel('Time in hours')
        plt.show()

    def call(self, inputs):
        inputs = tf.reshape(inputs, (-1, self.input_length, self.num_features))
        result = tf.py_function(self.numpy_curve_fit, [inputs], tf.float32)
        result = tf.reshape(result, (-1, self.out_steps, self.num_output_features))
        return result

    def numpy_curve_fit(self, inputs):
        """
        As this model does not work on tensors, a call to non-tensor function is done in call method. This method
        takes in tensor with batch_size and for every record in the batch fits and subsequently makes prediction
        that can be used for peak detection.

        :param inputs: tensor of shape (batch_size, input_length, num_features) First feature is kept, others are discarded
        :return: a tensor of shape (batch_size, out_steps, num_output_features(should be 1))
        """
        y_batch_data = tf.squeeze(inputs, axis=-1).numpy()
        x_data = np.arange(self.input_length) * 24
        output_data = []
        for y_data in y_batch_data:
            popt, _ = curve_fit(self.move_curve_function, x_data, y_data, p0=[0])
            x_fit = np.arange(self.input_length, self.input_length + self.out_steps) * 24
            y_fit = sin_function(x_fit, popt[0], self.period)
            noise = np.random.normal(0, self.noise, y_fit.shape)
            y_fit = y_fit + noise
            output_data.append(y_fit)
        stacked_array = np.stack(output_data)
        expanded_array = np.expand_dims(stacked_array, axis=-1)
        tensor = tf.convert_to_tensor(expanded_array)
        return np.array(tensor, dtype=np.float32)

    def move_curve_function(self, x_data, b):
        return sin_function(x_data, b, self.period)

    def get_peaks(self, prediction, method='raw'):
        """
        For given model predictions identifies peaks in it.

        :param prediction: ndarray of predictions (one feature)
        :param method: NA for this model
        :return: ndarray of indexes where peaks were detected in the input array
        """
        pred_peaks, _ = scipy.signal.find_peaks(prediction, distance=self.min_peak_distance)
        return pred_peaks


    def get_config(self):
        # Return the configuration of the model (needed for saving and loading)
        config = super().get_config().copy()
        config.update({
            "input_length": self.input_length,
            "out_steps": self.out_steps,
            "num_features": self.num_features,
            "min_peak_distance": self.min_peak_distance,
            "noise": self.noise,
            "period": self.period,
            "shift": self.shift
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)