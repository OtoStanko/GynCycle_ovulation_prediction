import scipy.signal
import tensorflow as tf

import IPython
import IPython.display

from ovulation_predicting.preprocessing_functions import create_classification_dataset

from .model_attention import Attention
from .model_classification import ClassificationMLP
from .model_cnn import WideCnn
from .model_cnn_lstm import CnnLstm
from .model_rnn import FeedBack


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


class MyModelWrapper:
    def __init__(self, features, input_width, out_steps, multi_window, loss_functions, max_epochs):
        self.features = features
        self.input_width = input_width
        self.out_steps = out_steps
        self.multi_window = multi_window
        self.loss_functions = loss_functions
        self.max_epochs = max_epochs

    def compile_and_fit(self, model_to_refactor, window, tensor_callback=None, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')
        history = None
        for loss in self.loss_functions:
            model_to_refactor.compile(loss=loss,
                                      optimizer=tf.keras.optimizers.Adam(),
                                      metrics=[tf.keras.metrics.MeanAbsoluteError()])
            if tensor_callback is not None:
                callbacks = [early_stopping, tensor_callback]
            else:
                callbacks = [early_stopping]
            history = model_to_refactor.fit(window.train, epochs=self.max_epochs,
                                            validation_data=window.val,
                                            callbacks=callbacks)
        return history

    def autoregressive_model(self):
        """
        # autoregressive RNN
        """
        feedback_model = FeedBack(32, self.out_steps, len(self.features), 20)
        prediction, state = feedback_model.warmup(self.multi_window.example[0])
        IPython.display.clear_output()
        #log_dir = "logs/fit/"
        #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        print(prediction.shape)
        print('Output shape (batch, time, features): ', feedback_model(self.multi_window.example[0]).shape)
        history = self.compile_and_fit(feedback_model, self.multi_window)
        return feedback_model

    def multistep_cnn(self):
        multi_cnn = WideCnn(self.input_width, self.out_steps, len(self.features), 20)
        IPython.display.clear_output()
        print('Output shape (batch, time, features): ', multi_cnn(self.multi_window.example[0]).shape)
        history = self.compile_and_fit(multi_cnn, self.multi_window)
        return multi_cnn

    def cnn_lstm(self, filters=None, ks=None, dilations=None):
        cnn_lstm_model = CnnLstm(16, self.input_width, self.out_steps, len(self.features), 20,
                                  filters, ks, dilations)
        IPython.display.clear_output()
        # print('Output shape (batch, time, features): ', cnn_lstm_model(multi_window.example[0]).shape)
        history = self.compile_and_fit(cnn_lstm_model, self.multi_window)
        return cnn_lstm_model

    def attention(self):
        attention_model = Attention(64, self.input_width, self.out_steps, len(self.features), 20)
        IPython.display.clear_output()
        history = self.compile_and_fit(attention_model, self.multi_window)
        return attention_model

    def classification_mlp(self, train_inputs, train_labels, val_inputs, val_labels, min_peak_distance=20):
        classification_model = ClassificationMLP(self.input_width, self.out_steps, 1, min_peak_distance)
        # log_dir = "logs/fit/"
        # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          mode='min')
        classification_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                                     optimizer=tf.keras.optimizers.Adam(),
                                     metrics=[tf.keras.metrics.CategoricalCrossentropy()])
        history = classification_model.fit(x=train_inputs, y=train_labels, validation_data=(val_inputs, val_labels),
                                           epochs=self.max_epochs, callbacks=[early_stopping], shuffle=True, batch_size=32)
        return classification_model

    def classification_datasets(self, train_df, val_df, test_df, features, feature_for_peaks, min_peak_height=0.3):
        # Dataset is normalized
        train_df_peaks, _ = scipy.signal.find_peaks(train_df[feature_for_peaks], distance=10, height=min_peak_height)
        val_df_peaks, _ = scipy.signal.find_peaks(val_df[feature_for_peaks], distance=10, height=min_peak_height)
        test_df_peaks, _ = scipy.signal.find_peaks(test_df[feature_for_peaks], distance=10, height=min_peak_height)

        train_inputs, train_labels = create_classification_dataset(
            train_df, features, train_df_peaks, self.input_width, self.out_steps)
        val_inputs, val_labels = create_classification_dataset(
            val_df, features, val_df_peaks, self.input_width, self.out_steps)
        test_inputs, test_labels = create_classification_dataset(
            test_df, features, test_df_peaks, self.input_width, self.out_steps)
        return train_inputs, train_labels, val_inputs, val_labels
