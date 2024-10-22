import pandas as pd
import plotly.graph_objs as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import scipy.signal

from Python_model.forecasting.custom_losses import Peak_loss
from Python_model.forecasting.models import FeedBack, WideCNN, ClassificationMLP
from preprocessing_functions import *


TRAIN_DATA_SUFFIX = '1_n'
workDir = os.path.join(os.getcwd(), "../outputDir/")
SAMPLING_FREQUENCY = 24
SAMPLING_FREQUENCY_UNIT = 'H'
NUM_INITIAL_DAYS_TO_DISCARD = 50
features = ['LH']
OUT_STEPS = 35
INPUT_WIDTH = 10
hormone = 'LH'


combined_df = create_dataframe(workDir, features, 'Time', TRAIN_DATA_SUFFIX)
combined_df['Time'] = combined_df['Time'] * 24
filtered_df = combined_df[combined_df['Time'] > NUM_INITIAL_DAYS_TO_DISCARD * 24]
filtered_df.set_index('Time', inplace=True)
sampled_df_timeH_index = [i for i in range(NUM_INITIAL_DAYS_TO_DISCARD * 24, int(filtered_df.index[-1]) + 1, SAMPLING_FREQUENCY)]
sampled_df_timeH = sample_data(filtered_df, sampled_df_timeH_index, features)

n = len(sampled_df_timeH)
train_df = sampled_df_timeH[0:int(n*0.7)]
val_df = sampled_df_timeH[int(n*0.7):int(n*0.9)]
test_df = sampled_df_timeH[int(n*0.9):]
train_df, norm_properties = normalize_df(train_df, method='minmax', values={feature: (0, 1) for feature in features})
val_df, _ = normalize_df(val_df, method='own', values=norm_properties)
test_df, _ = normalize_df(test_df, method='own', values=norm_properties)

save_models_dir = os.path.join(os.getcwd(), "../saved_models/")
saved_models_names = ['feed_back_RUN0_IN10', 'minPeakDist_24_RUN0_IN10', 'wide_cnn_RUN0_IN10']
saved_models_paths = [os.path.join(save_models_dir, model_name) for model_name in saved_models_names]
list_of_models = []
for model_name in saved_models_paths:
    model = tf.keras.models.load_model(model_name,
            custom_objects={'FeedBack': FeedBack, 'WideCNN': WideCNN,
                            'ClassificationMLP': ClassificationMLP, 'Peak_loss': Peak_loss})
    list_of_models.append(model)


df = test_df
df.index = (df.index - df.index[0]) / 24

peaks, _ = scipy.signal.find_peaks(df[hormone], distance=10, height=0.3)
plt.plot(df.index, df[hormone])
plt.scatter(df.index[peaks], df[hormone].iloc[peaks],
            color='red', zorder=5, label='LH Peaks')
plt.xlabel('Time [hours]')
plt.title('Test {} data'.format(hormone))
plt.show()


window_size = INPUT_WIDTH + OUT_STEPS
fig = make_subplots(rows=1, cols=1)

initial_window = df.iloc[:window_size]
trace = go.Scatter(
    x=initial_window.index,
    y=initial_window[hormone],
    mode='lines+markers',
    name='Sliding Window',
)
fig.add_trace(trace)

curr_peaks = peaks[peaks < window_size]
highlighted_values = df.loc[df.index.isin(peaks)]
highlighted_trace = go.Scatter(
    x=df.index[curr_peaks],
    y=df[hormone].iloc[curr_peaks],
    mode='markers',
    marker=dict(color='red', size=10),
    name='gt LH peaks',
)
fig.add_trace(highlighted_trace)

for model in list_of_models:
    window_data = df.iloc[:window_size]
    inputs = window_data[hormone].iloc[:10]
    tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
    reshaped_tensor = tf.reshape(tensor, (1, INPUT_WIDTH, 1))
    model_predictions = model(reshaped_tensor)
    predictions = tf.reshape(model_predictions, (1, OUT_STEPS, 1))
    predictions = predictions[0][:, 0]
    x = window_data.index[INPUT_WIDTH:]
    y = predictions.numpy()
    trace = go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        name=model._name,
    )
    fig.add_trace(trace)
    pred_peaks = model.get_peaks(predictions)
    offset_pred_peaks = pred_peaks + window_data.index[0] + INPUT_WIDTH
    y_peaks = y[pred_peaks]
    trace = go.Scatter(
        x=offset_pred_peaks,
        y=y_peaks,
        mode='markers',
        marker=dict(color='darkred'),
        showlegend=False,
    )
    fig.add_trace(trace)

fig.add_vline(
    x=initial_window.index[INPUT_WIDTH]-0.5,
    line=dict(color='red', width=2, dash='dash'),
)



# Update layout with sliders
steps = []
for i in range(len(df) - window_size + 1):
    window_data = df.iloc[i:i + window_size]
    #curr_peaks = peaks[i < peaks < i+window_size]
    curr_peaks = peaks[peaks < i+window_size]
    curr_peaks = curr_peaks[curr_peaks >= i]

    input_output_division = window_data.index[INPUT_WIDTH]

    x_values = [window_data.index, df.index[curr_peaks]]
    y_values = [window_data[hormone], df[hormone].iloc[curr_peaks]]

    for model in list_of_models:
        inputs = window_data[hormone].iloc[:10]
        tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
        reshaped_tensor = tf.reshape(tensor, (1, INPUT_WIDTH, 1))
        model_predictions = model(reshaped_tensor)
        predictions = tf.reshape(model_predictions, (1, OUT_STEPS, 1))
        predictions = predictions[0][:, 0]
        x_values.append(window_data.index[INPUT_WIDTH:])
        y = predictions.numpy()
        y_values.append(y)
        pred_peaks = model.get_peaks(predictions)
        offset_pred_peaks = pred_peaks + window_data.index[0] + INPUT_WIDTH
        y_peaks = y[pred_peaks]
        x_values.append(offset_pred_peaks)
        y_values.append(y_peaks)
    # Each step represents one window
    step = dict(
        method="update",
        args=[{
            'x': x_values,
            'y': y_values
        }, {
            'shapes': [
                {
                    'type': 'line',
                    'x0': input_output_division-0.5,
                    'y0': 0,
                    'x1': input_output_division-0.5,
                    'y1': 1,
                    'line': dict(color='red', width=2, dash='dash'),
                }
            ]
        }],
        label=f'Window {i + 1}'
    )
    steps.append(step)

# Create slider
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Window: "},
    pad={"t": 50},
    steps=steps,
)]


# Update figure layout to include the slider
fig.update_layout(
    sliders=sliders,
    title='Sliding Window Time Series Visualization',
    xaxis_title='Date',
    yaxis_title='{} levels'.format(hormone),
    width=800,
    height=400,
    yaxis=dict(range=[0, 1]),
)

# Show the interactive plot
fig.show()
