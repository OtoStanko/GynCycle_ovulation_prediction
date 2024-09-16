import numpy as np
import tensorflow as tf

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
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features, kernel_initializer=tf.initializers.he_normal())

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


class Wide_CNN(tf.keras.Model):
    def __init__(self, input_length, out_steps, num_features):
        super().__init__()
        self.input_length = input_length
        self.out_steps = out_steps
        self.num_features = num_features
        conv_model_wide = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32,
                                   kernel_size=(input_length,),
                                   activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=num_features, kernel_initializer=tf.initializers.he_normal()),
        ])
        self.cnn = conv_model_wide

    def call(self, inputs):
        #input = np.array(inputs)
        #input.append(inputs)
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        input_tensor = inputs
        for i in range(self.out_steps):
            input_data = input_tensor[:, -self.input_length:, :]
            #input_data = [input_tensor[0][-self.input_length+i] for i in range(self.input_length)]
            #input_data = tf.stack(input_data)
            #input_data = tf.expand_dims(input_data, axis=0)
            #input_data = input_tensor[-self.input_length:]
            y = self.cnn(input_data)
            input_tensor = tf.concat([input_tensor, y], axis=1)
        predictions = input_tensor[:, -self.out_steps:, :]
        #predictions = tf.stack(predictions)
        #predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
