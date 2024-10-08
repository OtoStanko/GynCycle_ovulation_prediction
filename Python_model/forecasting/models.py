import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from supporting_scripts import curve_function


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        # The prediction for each time step is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta


class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features, kernel_initializer=tf.initializers.he_normal())

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


class WideCNN(tf.keras.Model):
    def __init__(self, input_length, out_steps, num_features):
        super().__init__()
        self.input_length = input_length
        self.out_steps = out_steps
        self.num_features = num_features
        conv_model_wide = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=256,
                                   kernel_size=input_length-2,
                                   activation='relu',
                                   input_shape=(input_length, num_features),),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=num_features, kernel_initializer=tf.initializers.he_normal()),
        ])
        self.cnn = conv_model_wide

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        input_tensor = inputs
        for i in range(self.out_steps):
            input_data = input_tensor[:, -self.input_length:, :]
            y = self.cnn(input_data)
            input_tensor = tf.concat([input_tensor, y], axis=1)
        predictions = input_tensor[:, -self.out_steps:, :]
        return predictions


class NoisySinCurve(tf.keras.Model):
    def __init__(self, input_length, out_steps, num_features, train_df, feature, noise=0, period=25):
        super().__init__()
        self.input_length = input_length
        self.out_steps = out_steps
        self.num_features = num_features
        self.noise = noise / 10
        self.period = period
        x_data = train_df.index.values
        y_data = train_df[feature].values
        popt, _ = curve_fit(curve_function, x_data, y_data, p0=[1, 1, self.period])
        self.a_opt, self.b_opt, self.c_opt = popt
        print(f"Optimal parameters: a={self.a_opt}, b={self.b_opt}, c={self.c_opt}")
        print('a * sin(x * (2*pi/(c*24)) - b)')
        x_fit = np.linspace(1200, 3500, 100)
        y_fit = curve_function(x_fit, *popt)
        plt.plot(train_df.index[:100], train_df[feature][:100], color='black')
        plt.plot(x_fit, y_fit, label='Fitted Curve', color='orange')
        plt.title('Sampled dataframe with raw hours with fitted sin curve')
        plt.ylabel('Time in hours')
        plt.show()

    def call(self, inputs):
        inputs = tf.reshape(inputs, (self.input_length,))
        result = tf.py_function(self.numpy_curve_fit, [inputs], tf.float32)
        result = tf.reshape(result, (1, self.out_steps, self.num_features))
        return result

    def numpy_curve_fit(self, inputs):
        y_data = np.array(inputs)  # Convert TensorFlow tensor to NumPy array
        x_data = np.arange(self.input_length) * 24  # Create x_data array
        popt, _ = curve_fit(self.move_curve_function, x_data, y_data, p0=[self.b_opt])
        x_fit = np.arange(len(inputs), len(inputs) + self.out_steps) * 24
        y_fit = curve_function(x_fit, self.a_opt, popt[0], self.c_opt)
        noise = np.random.normal(0, self.noise, y_fit.shape)
        y_fit = y_fit + noise
        print(popt)
        return np.array(y_fit, dtype=np.float32)

    def move_curve_function(self, x_data, b):
        return curve_function(x_data, self.a_opt, b, self.c_opt)


class Distributed_peaks(tf.keras.Model):
    def __init__(self, out_steps, num_features, position):
        super().__init__()
        self.out_steps = out_steps
        self.num_features = num_features
        self.position = position  # Position of the first peak. 0 - out_steps-1

    def call(self, inputs):
        """
        Idea:
            Put one peak based on the position. Try to put another peak roughly 10 (11) days later.
            put som small values as a noise in between
        """
        result = np.array([random.random()/10 for _ in range(self.position-2)])
        result = np.append(result, 0.2)
        result = np.append(result, 0.5)
        result = np.append(result, 0.2)
        result = np.append(result, np.array([random.random()/10 for _ in range(self.out_steps-self.position-1)]))
        result = tf.reshape(result, (1, self.out_steps, self.num_features))
        return result

