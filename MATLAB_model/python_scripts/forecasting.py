from cProfile import label

import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.signal
import seaborn as sns
import numpy as np
import tensorflow as tf
#from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from windowGenerator import WindowGenerator
from models import Baseline, ResidualWrapper, FeedBack, Wide_CNN, My_rnn
import IPython
import IPython.display
from scipy.optimize import curve_fit
import supporting_scripts as sp

class Custom_loss(tf.keras.Loss):
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


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
    # MeanSquaredError(),
    # Huber()
    model.compile(loss=tf.keras.losses.Huber(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])
    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
    return history


def sample_data(original_df, new_index, columns):
    # The records are not evenly distributed. We will do sampling with linear interpolation for the models
    """
        for every time in the new index, find the largest smaller value and smallest larger value
        and interpolate them to get the new value
        Edge case if at least one of the is the same time
    """
    hormone_levels = {key: [] for key in columns}
    i = 0
    for curr_time in new_index:
        while original_df.index[i + 1] < curr_time:
            i += 1
        for feature in columns:
            # index_of_largest_smaller_time = i
            x0 = largest_smaller_time = original_df.index[i]
            y0 = largest_smaller_value = original_df[feature][original_df.index[i]]
            x1 = smallest_larger_time = original_df.index[i + 1]
            y1 = smallest_larger_value = original_df[feature][original_df.index[i + 1]]
            x = curr_time
            if type(curr_time) == np.datetime64:
                diff_x1_x = (x1 - x).total_seconds()
                diff_x_x0 = (x - x0).total_seconds()
                diff_x1_x0 = (x1 - x0).total_seconds()
            else:
                diff_x1_x = (x1 - x)
                diff_x_x0 = (x - x0)
                diff_x1_x0 = (x1 - x0)
            y = y0 * (diff_x1_x / diff_x1_x0) + y1 * (diff_x_x0 / diff_x1_x0)
            hormone_levels[feature].append(y)

    sampled_df = pd.DataFrame()
    for feature in columns:
        sampled_df[feature] = np.array(hormone_levels[feature])
    sampled_df.index = new_index
    sampled_df.index.name = 'DateTime'
    return sampled_df


def create_dataframe(input_files_directory, features, time_file_prefix, run_id=1):
    time_file = os.path.join(input_files_directory, "{}_{}.csv".format(time_file_prefix, run_id))
    times = pd.read_csv(time_file, header=None, names=[time_file_prefix])
    hormone_levels = [times]
    for feature in features:
        feature_file = os.path.join(input_files_directory, "{}_{}.csv".format(feature, run_id))
        feature_values = pd.read_csv(feature_file, header=None, names=[feature])
        hormone_levels.append(feature_values)
    combined_df = pd.concat(hormone_levels, axis=1)
    return combined_df


def compare_multiple_models(list_of_models, test_df, input_length, pred_length, features, hormone,
                            duration=250, step=5, plot=True, peak_comparison_distance=2):
    MIN_PEAK_DISTANCE = 20
    MIN_PEAK_HEIGHT = 0.3
    peaks, properties = scipy.signal.find_peaks(test_df[hormone], distance=MIN_PEAK_DISTANCE/2, height=MIN_PEAK_HEIGHT)
    if plot:
        plt.plot(test_df.index, test_df[hormone])
        plt.scatter(test_df.index[peaks], test_df[hormone].iloc[peaks],
                    color='red', zorder=5, label='Highlighted Points')
        plt.xlabel('Time [hours]')
        plt.title('Test {} data'.format(hormone))
        plt.show()
    model_peaks_mae = {}
    model_peaks_rmse = {}
    peaks_within_threshold = {}
    peaks_outside_threshold = {}
    sum_of_dists_to_nearest_peak = {}
    for offset in range(0, duration-pred_length-input_length+1, step):
        # Get the input data based on the offset and make a prediction
        inputs = []
        for feature in features:
            input = np.array(test_df[feature][offset:input_length + offset], dtype=np.float32)
            tensor = tf.convert_to_tensor(input, dtype=tf.float32)  # Ensure dtype is compatible
            inputs.append(tensor)
        tensor_inputs = tf.squeeze(inputs)
        reshaped_tensor = tf.reshape(tensor_inputs, (1, input_length, len(features)))
        list_of_model_predictions = []
        for model in list_of_models:
            predictions = model.predict(reshaped_tensor)
            predictions = predictions[0][:,0]
            list_of_model_predictions.append(predictions)
        # ground-truth time in days shifted to start with 0
        gt_time = test_df.index[offset:input_length + pred_length + offset]
        gt_time = gt_time / 24
        first_elem = gt_time[0]
        gt_time = gt_time - first_elem
        # prediction time in days shifted so that is starts with input_length th day
        pred_time = test_df.index[input_length + offset:input_length + pred_length + offset]
        pred_time = pred_time / 24
        pred_time = pred_time - first_elem
        #
        ground_truth = test_df[hormone][offset:input_length + pred_length + offset]
        # take only the peaks in the prediction window
        curr_peaks = np.array([x for x in peaks if offset <= x < input_length + pred_length + offset])
        curr_peaks = curr_peaks - offset
        # Try all the peaks, shift them to match the predicted data
        all_peaks_offset = np.array([x for x in peaks]) - offset
        if plot:
            plt.plot(gt_time, ground_truth, marker='.',)
        # Plot the tips of the peaks that are in the input-prediction window
        if len(curr_peaks) > 0 and plot:
            plt.scatter(gt_time[curr_peaks], ground_truth.iloc[curr_peaks],
                        color='red', zorder=5, label='Test data peaks')
        for i in range(len(list_of_model_predictions)):
            model = list_of_models[i]
            model_predictions = list_of_model_predictions[i]
            pred_peaks, properties = scipy.signal.find_peaks(model_predictions, distance=MIN_PEAK_DISTANCE)
            offset_pred_peaks = pred_peaks + input_length
            unfiltered_distances = sp.get_distances(all_peaks_offset, offset_pred_peaks)
            if len(curr_peaks) > 0:
                filtered_distances = np.array([distance for distance in unfiltered_distances if distance <= peak_comparison_distance])
                mae = np.mean(filtered_distances) if len(filtered_distances) > 0 else 0
                rmse = np.sqrt(np.mean(filtered_distances ** 2)) if len(filtered_distances) > 0 else 0
                model_peaks_mae[model._name] = model_peaks_mae.get(model._name, 0) + mae
                model_peaks_rmse[model._name] = model_peaks_rmse.get(model._name, 0) + rmse
                peaks_within_threshold[model._name] = (
                        peaks_within_threshold.get(model._name, 0) + len(filtered_distances))
                peaks_outside_threshold[model._name] = (
                        peaks_outside_threshold.get(model._name, 0) + len(pred_peaks) - len(filtered_distances))
                sum_of_dists_to_nearest_peak[model._name] = (
                        sum_of_dists_to_nearest_peak.get(model._name, 0) + sum(unfiltered_distances))
            if plot:
                line, = plt.plot(pred_time, model_predictions, marker='.', label=model._name)
                line_color = line.get_color()
                darker_line_color = sp.darken_color(line_color, 0.5)
                if len(unfiltered_distances) != 0:
                    for j in range(len(pred_peaks)):
                        if unfiltered_distances[j] <= peak_comparison_distance:
                            plt.scatter(pred_time[pred_peaks[j]], model_predictions[pred_peaks[j]],
                                        color='yellow', zorder=5)
                        else:
                            plt.scatter(pred_time[pred_peaks[j]], model_predictions[pred_peaks[j]],
                                        color=darker_line_color, zorder=5)
                else:
                    plt.scatter(pred_time[pred_peaks], model_predictions[pred_peaks],
                                color=darker_line_color, zorder=5)
        if plot:
            plt.axvline(x=input_length, color='r', linestyle='--',)
            plt.legend(loc='upper left')
            plt.title('Prediction on {} days with offset {} days'.format(input_length, offset))
            plt.show()
    print(model_peaks_mae)
    print(model_peaks_rmse)
    print(peaks_within_threshold)
    print(peaks_outside_threshold)
    return peaks_within_threshold, peaks_outside_threshold, sum_of_dists_to_nearest_peak


def test_model(model, test_df, train_df_mean, train_df_std, input_length, pred_length, hormone, duration=170, step=5):
    print(train_df_mean, train_df_std)
    test_df = (test_df - train_df_mean) / train_df_std
    plt.plot(test_df.index, test_df[hormone])
    plt.xlabel('Time [hours]')
    plt.title('Test {} data'.format(hormone))
    plt.show()
    for offset in range(0, duration-pred_length, step):
        input = np.array(test_df[hormone][offset:input_length + offset], dtype=np.float32)
        for i in range(pred_length):
            input_data = input[i:input_length + i]
            input_data = input_data.reshape(1, input_length, 1)
            y = model(input_data)[0][0]
            input = np.append(input, y)
        gt_time = test_df.index[offset:input_length + pred_length + offset]
        gt_time = gt_time / 24
        first_elem = gt_time[0]
        gt_time = gt_time - first_elem
        pred_time = test_df.index[input_length + offset:input_length + pred_length + offset]
        pred_time = pred_time / 24
        pred_time = pred_time - first_elem
        plt.plot(gt_time, test_df[hormone][offset:input_length + pred_length + offset])
        plt.plot(pred_time, input[input_length:input_length + pred_length + offset])
        plt.axvline(x=input_length, color='r', linestyle='--',)
        plt.title('Prediction on {} days with offset {} days'.format(input_length, offset))
        plt.show()
    # first 35 days are the base on which we are predicting one time step


def normalize_df(df, method='standard', values=None):
    """
    methods: standardization, mean and std may be provided, otherwise are calculated values=(mean, std) is expected
             minmax, if no values are provided, the scale to [0, 1] is done, otherwise to [a, b]
    """
    prop = {}
    if method == 'standard':
        for feature in df.columns:
            if values is None:
                df_mean = df[feature].mean()
                df_std = df[feature].std()
            else:
                df_mean, df_std = values[feature]
            df[feature] = (df[feature] - df_mean) / df_std
            prop[feature] = (df_mean, df_std)
    elif method == 'minmax':
        for feature in df.columns:
            min_val = np.min(df[feature])
            max_val = np.max(df[feature])
            if values is None:
                a = 0
                b = 1
            else:
                a, b = values[feature]
            df[feature] = a + ((df[feature] - min_val) * (b - a) / (max_val - min_val))
            prop[feature] = (min_val, max_val)
    elif method == 'own':
        for feature in df.columns:
            df[feature] = (df[feature] - values[feature][0]) / values[feature][1]
        prop = values
    return df, prop

# Set the parameters
workDir = os.path.join(os.getcwd(), "../outputDir/")
sampling_frequency = 24
sampling_frequency_unit = 'H'
num_initial_days_to_discard = 50
test_days_end = 300
#hormone = 'LH'
#features = ['FSH', 'E2', 'P4', 'LH']
features = ['LH']
MAX_EPOCHS = 25

# test on a small TS
test_dataframe = create_dataframe(workDir, features, 'Time', 1)
test_dataframe['Time'] = test_dataframe['Time'] * 24
# train on a long TS
combined_df = create_dataframe(workDir, features, 'Time', 4)
combined_df['Time'] = combined_df['Time'] * 24

print('Num records in the loaded data for training:', len(combined_df['Time']))
# Plot the loaded data
sns.set()
plt.ylabel('{} levels'.format('Hormones'))
plt.xticks(rotation=45)
plt.xlabel('Time in hours')
plt.plot(combined_df['Time'], combined_df[features], )
plt.title('Loaded combined dataframe for training')
plt.show()


# First 50 days of the simulation may be a bit messy and thus we ignore them
filtered_df = combined_df[combined_df['Time'] > num_initial_days_to_discard*24]
filtered_df.set_index('Time', inplace=True)

filtered_test_df = test_dataframe[test_dataframe['Time'] > num_initial_days_to_discard*24]
filtered_test_df.set_index('Time', inplace=True)

filtered_df_timeH = filtered_df.copy()


"""time_delta = pd.to_timedelta(filtered_df.index, unit='h')
datetime_index = start_date + time_delta
filtered_df.index = datetime_index
filtered_df.index.name = 'DateTime'
print(filtered_df[hormone][0])


# The records are not evenly distributed. We will do sampling with linear interpolation for the models
new_index = pd.date_range(start=data_start_date,
                          end=data_stop_date,
                          freq="{}{}".format(sampling_frequency, sampling_frequency_unit))

sampled_df = sample_data(filtered_df, new_index, features)

plt.plot(sampled_df.index, sampled_df[features], )
plt.title('Sampled dataframe with datetime')
plt.show()"""

print(test_days_end * 24 + 1)
print(filtered_df_timeH.index[-1])

sampled_df_timeH_index = [i for i in range(num_initial_days_to_discard * 24, int(filtered_df_timeH.index[-1]) + 1, sampling_frequency)]
print("Number of days in the training data:", len(sampled_df_timeH_index))
sampled_df_timeH = sample_data(filtered_df_timeH, sampled_df_timeH_index, features)
print('Num records in the sampled dataframe with raw hours: (Should be the same as the above number)', len(sampled_df_timeH.index))
plt.plot(sampled_df_timeH.index, sampled_df_timeH[features], )
plt.title('Sampled dataframe with raw hours')
plt.xlabel('Time in hours')
plt.show()

sampled_test_df_timeH_index = [i for i in range(num_initial_days_to_discard * 24, int(filtered_test_df.index[-1]) + 1, sampling_frequency)]
sampled_test_df = sample_data(filtered_test_df, sampled_test_df_timeH_index, features)
print("Number of days in the testing data:", len(sampled_test_df_timeH_index))


"""
days = [23, 24, 25, 26, 27, 28, 29, 30]
for day in days:
    sampled_df_timeH['{} days'.format(day)] = (np.sin((sampled_df_timeH.index * ((2 * np.pi / (day * 24))))) * 5) + 15
    plt.plot(sampled_df_timeH['{} days'.format(day)])
    plt.plot(sampled_df_timeH[hormone])
    plt.title('{} days cycle sin function'.format(day))
    plt.show()
"""
"""fft = tf.signal.rfft(sampled_df_timeH[hormone])
f_per_dataset = np.arange(0, len(fft))
n_samples_h = len(sampled_df_timeH[hormone])
print(n_samples_h)
hours_per_year = 24*365.2524
years_per_dataset = n_samples_h/(hours_per_year)
print(years_per_dataset)
f_per_year = f_per_dataset/years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.xscale('log')
plt.ylim(0, 25000)
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 365.2524/12, 365.2524], labels=['1/Year', '1/month', '1/day'])
_ = plt.xlabel('Frequency (log scale)')
plt.show()"""

column_indices = {name: i for i, name in enumerate(sampled_df_timeH.columns)}

n = len(sampled_df_timeH)
train_df = sampled_df_timeH[0:int(n*0.7)]
val_df = sampled_df_timeH[int(n*0.7):int(n*0.9)]
test_df = sampled_df_timeH[int(n*0.9):]

num_features = sampled_df_timeH.shape[1]
print("Num features", num_features)

train_mean = train_df.mean()
train_std = train_df.std()


train_df, norm_properties = normalize_df(train_df, method='minmax', values={feature: (0, 1) for feature in features})
# values = {feature: (0, properties[feature][1]) for feature in features}
val_df, _ = normalize_df(val_df, method='own', values=norm_properties)
test_df, _ = normalize_df(test_df, method='own', values=norm_properties)
"""
train_df, norm_properties = normalize_df(train_df, method='standard')
# values = {feature: (0, properties[feature][1]) for feature in features}
val_df, _ = normalize_df(val_df, method='standard', values=norm_properties)
test_df, _ = normalize_df(test_df, method='standard', values=norm_properties)
"""

for feature in features:
    plt.plot(train_df.index, train_df[feature], color='yellow')
    plt.plot(val_df.index, val_df[feature], color='blue')
    plt.plot(test_df.index, test_df[feature], color='red')
    plt.title('Sampled raw hours split {} levels normalized'.format(feature))
    plt.xlabel('Time in hours')
    plt.show()

#sp.fit_sin_curve(train_df, hormone, val_df, test_df, sampled_df_timeH)


# Window
w2 = WindowGenerator(input_width=34, label_width=1, shift=1,
                     train_df=train_df, val_df=val_df, test_df=test_df,
                     label_columns=features)
#print(w2)


print(w2.train.element_spec)
for example_inputs, example_labels in w2.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')

single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=features)


val_performance = {}
performance = {}
wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=features)


def baseline_model():
    """
    # Baseline model
    Returns the previous value
    """
    baseline = Baseline(label_index=column_indices[features[0]])

    baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                     metrics=[tf.keras.metrics.MeanAbsoluteError()])

    val_performance['Baseline'] = baseline.evaluate(single_step_window.val, return_dict=True)
    performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0, return_dict=True)

    #print(wide_window)
    wide_window.plot(features[0], 'Baseline model predictions', baseline)


def linear_model():
    """
    # Linear model
    One dense layer. Easily interpretable
    """
    linear = tf.keras.Sequential([
        tf.keras.layers.Dense(units=len(features))
    ])
    history = compile_and_fit(linear, single_step_window)

    val_performance['Linear'] = linear.evaluate(single_step_window.val, return_dict=True)
    performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0, return_dict=True)

    for feature in features:
        wide_window.plot(feature, 'Linear model predictions', linear)

    plt.bar(x = range(len(train_df.columns)),
            height=linear.layers[0].kernel[:,0].numpy())
    axis = plt.gca()
    axis.set_xticks(range(len(train_df.columns)))
    _ = axis.set_xticklabels(train_df.columns, rotation=90)
    plt.show()


def dense_model():
    """
    # Dense model
    Two hidden layers with relu activation functions
    """
    dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=len(features))
    ])
    history = compile_and_fit(dense, single_step_window)

    val_performance['Dense'] = dense.evaluate(single_step_window.val, return_dict=True)
    performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0, return_dict=True)
    for feature in features:
        wide_window.plot(feature, 'Dense model predictions', dense)


"""
# Multistep dense
"""
LABEL_WIDTH = 24
def cnn_model():
    """
    # CNN
    One convolutional layer with relu and one dense layer with relu activation functions
    Input width = 3
    """
    CONV_WIDTH = 3
    conv_window = WindowGenerator(input_width=CONV_WIDTH, label_width=1, shift=1,
                                  train_df=train_df, val_df=val_df, test_df=test_df,
                                  label_columns=features)
    print(conv_window)
    conv_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(CONV_WIDTH,),
                               activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=len(features)),
    ])
    history = compile_and_fit(conv_model, conv_window)

    IPython.display.clear_output()
    val_performance['Conv'] = conv_model.evaluate(conv_window.val, return_dict=True)
    performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0, return_dict=True)
    INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
    wide_conv_window = WindowGenerator( input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=1,
                                        train_df=train_df, val_df=val_df, test_df=test_df,
                                        label_columns=features)
    for feature in features:
        wide_conv_window.plot(feature, 'CNN model predictions', conv_model)


def wide_cnn(width=35):
    """
    # CNN Wide window
    One convolutional layer with relu and one dense layer with relu activation functions
    Input width = 35
    """
    NUM_DAYS = width
    CONV_WIDTH_WIDE = NUM_DAYS
    conv_window_wide = WindowGenerator(input_width=CONV_WIDTH_WIDE, label_width=1, shift=1,
                                  train_df=train_df, val_df=val_df, test_df=test_df,
                                  label_columns=features)
    print(conv_window_wide)
    conv_model_wide = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(CONV_WIDTH_WIDE,),
                               activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=len(features)),
    ])
    history = compile_and_fit(conv_model_wide, conv_window_wide)

    IPython.display.clear_output()
    val_performance['Conv_wide'] = conv_model_wide.evaluate(conv_window_wide.val, return_dict=True)
    performance['Conv_wide'] = conv_model_wide.evaluate(conv_window_wide.test, verbose=0, return_dict=True)

    LABEL_WIDTH_WIDE = 24
    INPUT_WIDTH_WIDE = LABEL_WIDTH + (CONV_WIDTH_WIDE - 1)
    wide_conv_window = WindowGenerator( input_width=INPUT_WIDTH_WIDE, label_width=LABEL_WIDTH_WIDE, shift=1,
                                        train_df=train_df, val_df=val_df, test_df=test_df,
                                        label_columns=features)

    for feature in features:
        wide_conv_window.plot(feature, 'CNN wide model predictions', conv_model_wide)

    prediction_length = 35
    #test_model(conv_model_wide, sampled_test_df, train_mean, train_std, NUM_DAYS, prediction_length, hormone)


def lstm_model():
    """
    # RNN model
    # LSTM long short-term memory
    """
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=len(features))
    ])
    history = compile_and_fit(lstm_model, wide_window)

    IPython.display.clear_output()
    val_performance['LSTM'] = lstm_model.evaluate(wide_window.val, return_dict=True)
    performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0, return_dict=True)

    for feature in features:
        wide_window.plot(feature, 'RNN LSTM model predictions', lstm_model)


def residual_connections_model():
    """
    # Residual connections
    lstm model
    """
    residual_lstm = ResidualWrapper(
        tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(
            len(features),
            # The predicted deltas should start small.
            # Therefore, initialize the output layer with zeros.
            kernel_initializer=tf.initializers.zeros())
    ]))

    history = compile_and_fit(residual_lstm, wide_window)

    IPython.display.clear_output()
    val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val, return_dict=True)
    performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test, verbose=0, return_dict=True)

    for feature in features:
        wide_window.plot(feature, 'Residual LSTM model predictions', residual_lstm)


def show_performance():
    """
    # Performance
    """
    x = np.arange(len(performance))
    width = 0.3
    metric_name = 'mean_absolute_error'
    val_mae = [v[metric_name] for v in val_performance.values()]
    test_mae = [v[metric_name] for v in performance.values()]

    plt.ylabel('mean_absolute_error {}, normalized'.format(features))
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=performance.keys(),
               rotation=45)
    _ = plt.legend()
    plt.show()

    for name, value in performance.items():
        print(f'{name:12s}: {value[metric_name]:0.4f}')


#baseline_model()
#linear_model()
#dense_model()
#cnn_model()
#wide_cnn()
#residual_connections_model()
#show_performance()
print("Ok up to here")

"""
# Multi-step models
"""
OUT_STEPS = 35
INPUT_WIDTH = 35
multi_window = WindowGenerator(input_width=INPUT_WIDTH, label_width=OUT_STEPS,   shift=OUT_STEPS,
                               train_df=train_df, val_df=val_df, test_df=test_df,
                               label_columns=features)
for feature in features:
    multi_window.plot(feature, 'Multi window')

# variant with the artificially added samples
"""OUT_STEPS_a = 24
multi_window_a = WindowGenerator(input_width=24*24, label_width=OUT_STEPS_a*24,   shift=OUT_STEPS_a*24,
                               train_df=train_df_a, val_df=val_df_a, test_df=test_df_a,
                               label_columns=features)
multi_window_a.plot(hormone)"""

multi_val_performance = dict()
multi_performance = dict()


def autoregressive_model():
    """
    # autoregressive RNN
    """
    feedback_model = FeedBack(32, OUT_STEPS, len(features))
    prediction, state = feedback_model.warmup(multi_window.example[0])
    print(prediction.shape)
    print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)
    history = compile_and_fit(feedback_model, multi_window)

    IPython.display.clear_output()

    multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val, return_dict=True)
    multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0, return_dict=True)
    for feature in features:
        multi_window.plot(feature, 'Autoregressive model predictions', feedback_model)
    return feedback_model


def multistep_cnn():
    multi_cnn = Wide_CNN(INPUT_WIDTH, OUT_STEPS, len(features))
    IPython.display.clear_output()
    print('Output shape (batch, time, features): ', multi_cnn(multi_window.example[0]).shape)
    history = compile_and_fit(multi_cnn, multi_window)

    multi_val_performance['CNN'] = multi_cnn.evaluate(multi_window.val, return_dict=True)
    multi_performance['CNN'] = multi_cnn.evaluate(multi_window.test, verbose=0, return_dict=True)
    for feature in features:
        multi_window.plot(feature, 'CNN model predictions', multi_cnn)
    return multi_cnn


def more_layers_rnn():
    mlr = My_rnn(32, OUT_STEPS, len(features))
    IPython.display.clear_output()
    print('Output shape (batch, time, features): ', mlr(multi_window.example[0]).shape)
    history = compile_and_fit(mlr, multi_window)

    multi_val_performance['drnn'] = mlr.evaluate(multi_window.val, return_dict=True)
    multi_performance['drnn'] = mlr.evaluate(multi_window.test, verbose=0, return_dict=True)
    for feature in features:
        multi_window.plot(feature, 'CNN model predictions', mlr)
    return mlr


def multistep_performance():
    # Performance
    x = np.arange(len(multi_performance))
    width = 0.3

    metric_name = 'mean_absolute_error'
    val_mae = [v[metric_name] for v in multi_val_performance.values()]
    test_mae = [v[metric_name] for v in multi_performance.values()]

    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=multi_performance.keys(),
               rotation=45)
    plt.ylabel(f'MAE (average over all times and outputs)')
    _ = plt.legend()
    plt.show()


from collections import Counter
peaks, properties = scipy.signal.find_peaks(train_df[features[0]], distance=10, height=0.3)
distances = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
count = Counter(distances)
print("Number of cycles:", len(distances))
numbers = list(count.keys())
frequencies = list(count.values())
plt.bar(numbers, frequencies, color='skyblue')
plt.show()

sampled_test_df, _ = normalize_df(sampled_test_df, method='own', values=norm_properties)
peaks_within_threshold = {}
peaks_outside_threshold = {}
sum_of_dists_to_nearest_peak = {}
PEAK_COMPARISON_DISTANCE = 2
for _ in range(20):
    #feedback_model = autoregressive_model()
    #feedback_model._name = 'feed_back'
    multi_cnn_model = multistep_cnn()
    multi_cnn_model._name = 'wide_cnn'
    mrnn = more_layers_rnn()
    mrnn._name = 'drnn'
    within, outside, nearest_dists = compare_multiple_models([multi_cnn_model, mrnn],
                                              sampled_test_df, INPUT_WIDTH, OUT_STEPS, features, features[0], plot=False,
                                              peak_comparison_distance=PEAK_COMPARISON_DISTANCE)
    for model_name, value in within.items():
        peaks_within_threshold[model_name] = peaks_within_threshold.get(model_name, 0) + value
    for model_name, value in outside.items():
        peaks_outside_threshold[model_name] = peaks_outside_threshold.get(model_name, 0) + value
    for model_name, value in nearest_dists.items():
        sum_of_dists_to_nearest_peak[model_name] = sum_of_dists_to_nearest_peak.get(model_name, 0) + value
print(peaks_within_threshold)
print(peaks_outside_threshold)
sp.print_peak_statistics(peaks_within_threshold, peaks_outside_threshold, sum_of_dists_to_nearest_peak,
                         PEAK_COMPARISON_DISTANCE)
multistep_performance()
"""
sampled_train_df = sampled_df[(sampled_df.index > data_start_date) & (sampled_df.index <= data_tt_split_date)]
sampled_test_df = sampled_df[(sampled_df.index > data_tt_split_date) & (sampled_df.index <= data_stop_date)]

plt.plot(sampled_train_df.index, sampled_train_df[hormone], color = "black")
plt.plot(sampled_test_df.index, sampled_test_df[hormone], color = "red")
plt.show()
"""

"""
# Fit the model
order = (1,1,0)
seasonal_order = (1,1,0,36)
mod = SARIMAX(sampled_train_df[hormone], trend='c', order=order, seasonal_order=seasonal_order,
              freq="{}{}".format(sampling_frequency, sampling_frequency_unit))
res = mod.fit(disp=False)
print(res.summary())
"""
"""
print(train_df[hormone])
arima_model = ARIMA(train_df[hormone], order=(4,2,0), seasonal_order=(4,2,0,12),)
fitted_arima = arima_model.fit()
arima_predictions = fitted_arima.get_forecast(35)
arima_predictions_series = arima_predictions.predicted_mean
#print(arima_predictions_series)
plt.plot(arima_predictions_series.index, arima_predictions_series.values)
plt.show()
"""