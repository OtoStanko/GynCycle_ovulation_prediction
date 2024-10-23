import plotly.graph_objs as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import scipy.signal

from preprocessing_functions import *




class TimeSeriesVisualizer:
    def __init__(self, df, hormone, input_length, output_length):
        self.df = df
        self.hormone = hormone
        self.INPUT_LENGTH = input_length
        self.OUTPUT_LENGTH = output_length
        self.window_size = input_length + output_length
        self.steps = []
        self.batch_size = 32
        self.fig = make_subplots(rows=1, cols=1)

        self.df.index = (self.df.index - self.df.index[0]) / 24
        initial_window = df.iloc[:self.window_size]
        trace = go.Scatter(
            x=initial_window.index,
            y=initial_window[hormone],
            mode='lines+markers',
            name='Sliding Window',
        )
        self.fig.add_trace(trace)
        peaks, _ = scipy.signal.find_peaks(df[hormone], distance=10, height=0.3)
        curr_peaks = peaks[peaks < self.window_size]
        highlighted_trace = go.Scatter(
            x=df.index[curr_peaks],
            y=df[hormone].iloc[curr_peaks],
            mode='markers',
            marker=dict(color='red', size=10),
            name='gt LH peaks',
        )
        self.peaks = peaks
        self.fig.add_trace(highlighted_trace)

    def update_sliders(self, list_of_models=None):
        if list_of_models is not None:
            for model in list_of_models:
                window_data = self.df.iloc[:self.window_size]
                inputs = window_data[self.hormone].iloc[:self.INPUT_LENGTH]
                tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
                reshaped_tensor = tf.reshape(tensor, (1, self.INPUT_LENGTH, 1))
                model_predictions = model(reshaped_tensor)
                predictions = tf.reshape(model_predictions, (1, self.OUTPUT_LENGTH, 1))
                predictions = predictions[0][:, 0]
                x = window_data.index[self.INPUT_LENGTH:]
                y = predictions.numpy()
                trace = go.Scatter(
                    x=x,
                    y=y,
                    mode='lines+markers',
                    name=model._name,
                )
                self.fig.add_trace(trace)
                pred_peaks = model.get_peaks(predictions)
                offset_pred_peaks = pred_peaks + window_data.index[0] + self.INPUT_LENGTH
                y_peaks = y[pred_peaks]
                trace = go.Scatter(
                    x=offset_pred_peaks,
                    y=y_peaks,
                    mode='markers',
                    marker=dict(color='darkred'),
                    showlegend=False,
                )
                self.fig.add_trace(trace)
            self.fig.add_vline(
                x=self.df.index[self.INPUT_LENGTH] - 0.5,
                line=dict(color='red', width=2, dash='dash'),
            )
        i = 0
        limit = len(self.df) - self.window_size + 1
        while i < limit:
            current_batch_size = min(self.batch_size, limit - i)
            batch_data = [self.df.iloc[i + j:i + j + self.INPUT_LENGTH][self.hormone] for j in
                          range(current_batch_size)]
            if list_of_models is not None:
                tensor_batch = tf.convert_to_tensor(batch_data, dtype=tf.float32)
                reshaped_tensor_batch = tf.reshape(tensor_batch, (current_batch_size, self.INPUT_LENGTH, 1))
                batch_predictions_dict = {model._name: None for model in list_of_models}
                for model in list_of_models:
                    batch_predictions = model(reshaped_tensor_batch)
                    batch_predictions = tf.reshape(batch_predictions, (current_batch_size, self.OUTPUT_LENGTH, 1))
                    batch_predictions_dict[model._name] = batch_predictions

            for j in range(current_batch_size):
                window_data = self.df.iloc[i + j:i + j + self.window_size]
                curr_peaks = self.peaks[self.peaks < i + j + self.window_size]
                curr_peaks = curr_peaks[curr_peaks >= i + j]
                input_output_division = window_data.index[self.INPUT_LENGTH]
                x_values = [window_data.index, self.df.index[curr_peaks]]
                y_values = [window_data[self.hormone], self.df[self.hormone].iloc[curr_peaks]]
                # Each step represents one window
                args = [{
                        'x': x_values,
                        'y': y_values
                    }]
                if list_of_models is not None:
                    for model in list_of_models:
                        predictions = batch_predictions_dict[model._name][j][:, 0]
                        x_values.append(window_data.index[self.INPUT_LENGTH:])
                        y = predictions.numpy()
                        y_values.append(y)
                        pred_peaks = model.get_peaks(predictions)
                        offset_pred_peaks = pred_peaks + window_data.index[0] + self.INPUT_LENGTH
                        y_peaks = y[pred_peaks]
                        x_values.append(offset_pred_peaks)
                        y_values.append(y_peaks)
                    args.append({'shapes': [
                            {
                                'type': 'line',
                                'x0': input_output_division-0.5,
                                'y0': 0,
                                'x1': input_output_division-0.5,
                                'y1': 1,
                                'line': dict(color='red', width=2, dash='dash'),
                            }]
                    })
                step = dict(
                    method="update",
                    args=args,
                    label=f'Window {i + j + 1}'
                )
                self.steps.append(step)
            i += current_batch_size

    def show(self):
        self.sliders = [dict(
            active=0,
            currentvalue={"prefix": "Window: "},
            pad={"t": 50},
            steps=self.steps,
        )]
        self.fig.update_layout(
            sliders=self.sliders,
            title='Sliding Window Time Series Visualization',
            xaxis_title='Date',
            yaxis_title='{} levels'.format(self.hormone),
            width=800,
            height=400,
            yaxis=dict(range=[0, 1]),
        )
        self.fig.show()