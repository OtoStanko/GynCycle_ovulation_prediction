import tensorflow as tf
from keras.losses import Loss


class Custom_loss(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold = 0.3

    def call(self, y_true, y_pred):
        squared_error = 0.5 * tf.square(y_true - y_pred)
        quadruped_error = 2.0 * tf.square(y_true - y_pred) ** 2
        absolute_error = tf.abs(y_true - y_pred)
        mask = tf.greater_equal(y_true, self.threshold)
        mask2 = tf.greater_equal(y_true, 0.5)
        #loss = tf.where(mask, squared_error, absolute_error)
        loss = tf.where(mask2, quadruped_error,
                        tf.where(mask, squared_error, absolute_error))
        return tf.reduce_mean(loss)


class Peak_loss(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def find_peaks(self, tensor, height):
        # Shift tensor left and right to find peaks
        tensor = tf.squeeze(tensor, axis=-1)
        shifted_right = tf.roll(tensor, shift=1, axis=1)
        shifted_left = tf.roll(tensor, shift=-1, axis=1)

        peak_strength = tf.nn.relu(tensor - shifted_left) * tf.nn.relu(tensor - shifted_right)
        min_val = tf.reduce_min(peak_strength)
        max_val = tf.reduce_max(peak_strength)
        normalized_tensor = (peak_strength - min_val) / (max_val - min_val)
        peaks = tensor * (normalized_tensor + 0.01)
        peak_indicator = tf.sigmoid(peaks)
        peak_indicator = tf.expand_dims(peak_indicator, axis=-1)
        return peak_indicator

    def call(self, y_true, y_pred):
        MIN_PEAK_DISTANCE = 20
        MIN_PEAK_HEIGHT = 0.3
        ALPHA = 0.5

        true_peaks = self.find_peaks(y_true, MIN_PEAK_HEIGHT)
        pred_peaks = self.find_peaks(y_pred, MIN_PEAK_HEIGHT)

        mse = tf.keras.losses.MeanSquaredError()
        loss = ALPHA * mse(true_peaks, pred_peaks) + (1-ALPHA) * mse(y_true, y_pred)

        return loss