import plotly.graph_objs as go
import scipy.signal
from plotly.subplots import make_subplots

from preprocessing_functions import *


class TimeSeriesVisualizer:
    def __init__(self, df, hormones, input_length, output_length):
        self.df = df
        self.hormones = hormones
        self.num_features = len(hormones)
        self.INPUT_LENGTH = input_length
        self.OUTPUT_LENGTH = output_length
        self.window_size = input_length + output_length
        self.steps = []
        self.batch_size = 32
        self.fig = make_subplots(rows=1, cols=1)
        self.hoi_index = 0

        initial_window = df.iloc[:self.window_size]
        initial_window.index = (initial_window.index - initial_window.index[0]) / 24
        if len(hormones) == 0:
            print("No hormones defined")
            return
        self.peaks = {hormone: [] for hormone in hormones}
        for hormone in hormones:
            trace = go.Scatter(
                x=initial_window.index,
                y=initial_window[hormone],
                mode='lines+markers',
                name=hormone,
            )
            self.fig.add_trace(trace)
        hormone = hormones[self.hoi_index]
        peaks, _ = scipy.signal.find_peaks(df[hormone], distance=10, height=0.3)
        curr_peaks = peaks[peaks < self.window_size]
        highlighted_trace = go.Scatter(
            x=initial_window.index[curr_peaks],
            y=initial_window[hormone].iloc[curr_peaks],
            mode='markers',
            marker=dict(color='red', size=10),
            name='gt {} peaks'.format(hormones[self.hoi_index]),
        )
        self.peaks[hormone] = peaks
        self.fig.add_trace(highlighted_trace)

    def update_sliders(self, list_of_models=None):
        if len(self.hormones) == 0:
            print("No hormones defined")
            return
        if list_of_models is not None:
            window_data = self.df.iloc[:self.window_size]
            window_data.index = (window_data.index - window_data.index[0]) / 24
            input_data = []
            for hormone in self.hormones:
                feature_inputs = window_data[hormone].iloc[:self.INPUT_LENGTH]
                input_data.append(feature_inputs)
            tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
            reshaped_tensor = tf.reshape(tensor, (1, self.INPUT_LENGTH, self.num_features))
            for model in list_of_models:
                model_predictions = model(reshaped_tensor)
                predictions = tf.reshape(model_predictions, (1, self.OUTPUT_LENGTH, self.num_features))
                predictions = predictions[0][:, self.hoi_index]
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
                x=window_data.index[self.INPUT_LENGTH] - 0.5,
                line=dict(color='red', width=2, dash='dash'),
            )
        i = 0
        limit = len(self.df) - self.window_size + 1
        while i < limit:
            current_batch_size = min(self.batch_size, limit - i)
            if list_of_models is not None:
                batch_data = [self.df.iloc[i + j:i + j + self.INPUT_LENGTH][self.hormones].values for j in
                              range(current_batch_size)]
                tensor_batch = tf.convert_to_tensor(batch_data, dtype=tf.float32)
                reshaped_tensor_batch = tf.reshape(tensor_batch, (current_batch_size, self.INPUT_LENGTH, self.num_features))
                batch_predictions_dict = {model._name: None for model in list_of_models}
                for model in list_of_models:
                    batch_predictions = model(reshaped_tensor_batch)
                    batch_predictions = tf.reshape(batch_predictions, (current_batch_size, self.OUTPUT_LENGTH, self.num_features))
                    batch_predictions_dict[model._name] = batch_predictions

            for j in range(current_batch_size):
                x_values = []
                y_values = []
                window_data = self.df.iloc[i + j:i + j + self.window_size]
                window_data.index = (window_data.index - window_data.index[0]) / 24
                input_output_division = window_data.index[self.INPUT_LENGTH]
                for hormone in self.hormones:
                    x_values += [window_data.index]
                    y_values += [window_data[hormone]]
                hormone = self.hormones[self.hoi_index]
                curr_peaks = self.peaks[hormone][self.peaks[hormone] < i + j + self.window_size]
                curr_peaks = curr_peaks[curr_peaks >= i + j] - j - i
                x_values += [window_data.index[curr_peaks]]
                y_values += [ window_data[hormone].iloc[curr_peaks]]
                args = [{
                    'x': x_values,
                    'y': y_values
                }]
                if list_of_models is not None:
                    for model in list_of_models:
                        predictions = batch_predictions_dict[model._name][j][:, self.hoi_index]
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
            yaxis_title='{} levels'.format(self.hormones),
            width=800,
            height=400,
            yaxis=dict(range=[0, 1]),
        )
        self.fig.show()