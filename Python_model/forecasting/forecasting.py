import itertools

from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal
import seaborn as sns
import tensorflow as tf
import IPython
import IPython.display

from windowGenerator import WindowGenerator
from models import FeedBack, WideCNN, NoisySinCurve, Distributed_peaks
import supporting_scripts as sp
from custom_losses import Peak_loss
from preprocessing_functions import *
from model_comparison import ModelComparator


"""
    Parameters
"""
TRAIN_DATA_SUFFIX = 5
TEST_DATA_SUFFIX = 1
LOSS_FUNCTIONS = [tf.keras.losses.MeanSquaredError(), Peak_loss()]

# Set the parameters
workDir = os.path.join(os.getcwd(), "../outputDir/")
SAMPLING_FREQUENCY = 24
SAMPLING_FREQUENCY_UNIT = 'H'
NUM_INITIAL_DAYS_TO_DISCARD = 50
test_days_end = 300
features = ['LH']
MAX_EPOCHS = 25

NUM_RUNS = 1
PEAK_COMPARISON_DISTANCE = 2
PLOT_TESTING = True


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
    # tf.keras.losses.MeanSquaredError(),
    # tf.keras.losses.Huber()
    # Peak_loss()
    history = None
    for loss in LOSS_FUNCTIONS:
        model.compile(loss=loss,
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])
        history = model.fit(window.train, epochs=MAX_EPOCHS,
                          validation_data=window.val,
                          callbacks=[early_stopping])
    return history


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


# test on a small TS
test_dataframe = create_dataframe(workDir, features, 'Time', TEST_DATA_SUFFIX)
test_dataframe['Time'] = test_dataframe['Time'] * 24
# train on a long TS
combined_df = create_dataframe(workDir, features, 'Time', TRAIN_DATA_SUFFIX)
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
filtered_df = combined_df[combined_df['Time'] > NUM_INITIAL_DAYS_TO_DISCARD * 24]
filtered_df.set_index('Time', inplace=True)

filtered_test_df = test_dataframe[test_dataframe['Time'] > NUM_INITIAL_DAYS_TO_DISCARD * 24]
filtered_test_df.set_index('Time', inplace=True)

filtered_df_timeH = filtered_df.copy()


print(test_days_end * 24 + 1)
print(filtered_df_timeH.index[-1])

sampled_df_timeH_index = [i for i in range(NUM_INITIAL_DAYS_TO_DISCARD * 24, int(filtered_df_timeH.index[-1]) + 1, SAMPLING_FREQUENCY)]
print("Number of days in the training data:", len(sampled_df_timeH_index))
sampled_df_timeH = sample_data(filtered_df_timeH, sampled_df_timeH_index, features)
print('Num records in the sampled dataframe with raw hours: (Should be the same as the above number)', len(sampled_df_timeH.index))
plt.plot(sampled_df_timeH.index, sampled_df_timeH[features], )
plt.title('Sampled dataframe with raw hours')
plt.xlabel('Time in hours')
plt.show()

sampled_test_df_timeH_index = [i for i in range(NUM_INITIAL_DAYS_TO_DISCARD * 24, int(filtered_test_df.index[-1]) + 1, SAMPLING_FREQUENCY)]
sampled_test_df = sample_data(filtered_test_df, sampled_test_df_timeH_index, features)
print("Number of days in the testing data:", len(sampled_test_df_timeH_index))


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


def learn_model(model, window, features, val_performance, performance):
    history = compile_and_fit(model, window)
    val_performance[model._name] = model.evaluate(single_step_window.val, return_dict=True)
    performance[model._name] = model.evaluate(single_step_window.test, verbose=0, return_dict=True)

    for feature in features:
        window.plot(feature, model._name + 'Model predictions', model)



def show_performance(val_performance, performance):
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
    multi_cnn = WideCNN(INPUT_WIDTH, OUT_STEPS, len(features))
    IPython.display.clear_output()
    print('Output shape (batch, time, features): ', multi_cnn(multi_window.example[0]).shape)
    history = compile_and_fit(multi_cnn, multi_window)

    multi_val_performance['CNN'] = multi_cnn.evaluate(multi_window.val, return_dict=True)
    multi_performance['CNN'] = multi_cnn.evaluate(multi_window.test, verbose=0, return_dict=True)
    for feature in features:
        multi_window.plot(feature, 'CNN model predictions', multi_cnn)
    return multi_cnn


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
num_detected_peaks = {}
#tf.config.run_functions_eagerly(True)
model_comparator = ModelComparator(sampled_test_df, INPUT_WIDTH, OUT_STEPS, features, features[0],
                                   plot=PLOT_TESTING, peak_comparison_distance=PEAK_COMPARISON_DISTANCE)
for run_id in range(NUM_RUNS):
    feedback_model = autoregressive_model()
    feedback_model._name = 'feed_back'
    peak_start = Distributed_peaks(OUT_STEPS, len(features), 2)
    peak_start._name = 'peak_beginning'
    peak_middle = Distributed_peaks(OUT_STEPS, len(features), 16)
    peak_middle._name = 'peak_middle'
    peak_end = Distributed_peaks(OUT_STEPS, len(features), 30)
    peak_end._name = 'peak_end'
    multi_cnn_model = multistep_cnn()
    multi_cnn_model._name = 'wide_cnn'
    fitted_sin = NoisySinCurve(INPUT_WIDTH, OUT_STEPS, len(features), train_df, features[0], noise=0.1)
    fitted_sin._name = 'sin_curve'

    list_of_models = [feedback_model, fitted_sin, multi_cnn_model]
    model_comparator.compare_models(list_of_models, run_id)
    within, outside, nearest_dists, num_detected, peak_distances_distribution = model_comparator.get_run_results_tuple(run_id)

    for model_name, num_peaks_within in within.items():
        peaks_within_threshold[model_name] = peaks_within_threshold.get(model_name, list()) + [num_peaks_within]
    for model_name, num_peaks_outside in outside.items():
        peaks_outside_threshold[model_name] = peaks_outside_threshold.get(model_name, list()) + [num_peaks_outside]
    for model_name, nearest_peak_dist in nearest_dists.items():
        sum_of_dists_to_nearest_peak[model_name] = sum_of_dists_to_nearest_peak.get(model_name, list()) + [nearest_peak_dist]
    for model_name, num_detected_peak in num_detected.items():
        num_detected_peaks[model_name] = num_detected_peaks.get(model_name, list()) + [num_detected_peak]
    model_comparator.plot_peak_distances(run_id)
print(peaks_within_threshold)
print(peaks_outside_threshold)
sp.print_peak_statistics(peaks_within_threshold, peaks_outside_threshold, sum_of_dists_to_nearest_peak,
                         PEAK_COMPARISON_DISTANCE)
multistep_performance()

colors = mpl.colormaps.get_cmap('tab10')  # Using tab10 colormap with as many colors as there are keys

plt.figure(figsize=(8, 6))
for idx, key in enumerate(peaks_within_threshold):
    x_values = peaks_outside_threshold[key]
    y_values = peaks_within_threshold[key]

    # Plot each key's data with a unique color and label it with the key
    plt.scatter(x_values, y_values, color=colors(idx), label=key)

all_values = itertools.chain(*peaks_outside_threshold.values(), *peaks_within_threshold.values())
max_val = max(all_values) + 5
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.plot([0, max_val], [0, max_val], 'r--', label='y=x')
plt.xlabel('Num peaks outside the threshold')
plt.ylabel('Num peaks inside threshold')
plt.legend(title="Model")
plt.show()
