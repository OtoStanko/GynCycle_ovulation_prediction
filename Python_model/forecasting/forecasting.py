from collections import Counter

import IPython
import IPython.display
import matplotlib.pyplot as plt
import scipy.signal
import seaborn as sns

from ModelComparator import ModelComparator
from TimeSeriesVisualizer import TimeSeriesVisualizer
from custom_losses import Peak_loss
from models import FeedBack, WideCNN, ClassificationMLP, NoisySinCurve
from preprocessing_functions import *
from windowGenerator import WindowGenerator

"""
    Parameters
"""
TRAIN_DATA_SUFFIX = '1_n'
TEST_DATA_SUFFIX = 'of_1'
LOSS_FUNCTIONS = [tf.keras.losses.MeanSquaredError()]

# Set the parameters
inputDir = os.path.join(os.getcwd(), "../outputDir/")
save_models_dir = os.path.join(os.getcwd(), "../saved_models/")
SAMPLING_FREQUENCY = 24
SAMPLING_FREQUENCY_UNIT = 'H'
NUM_INITIAL_DAYS_TO_DISCARD = 50
features = ['LH', 'E2']
MAX_EPOCHS = 25

# forecast parameters
INPUT_WIDTH = 35
OUT_STEPS = 35

NUM_RUNS = 5
PEAK_COMPARISON_DISTANCE = 2
PLOT_TESTING = False
SAVE_MODELS = False


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


# test on a small TS
test_dataframe = create_dataframe(inputDir, features, 'Time', TEST_DATA_SUFFIX)
test_dataframe['Time'] = test_dataframe['Time'] * 24
# train on a long TS
combined_df = create_dataframe(inputDir, features, 'Time', TRAIN_DATA_SUFFIX)
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


print(filtered_df.index[-1])

sampled_df_timeH_index = [i for i in range(NUM_INITIAL_DAYS_TO_DISCARD * 24, int(filtered_df.index[-1]) + 1, SAMPLING_FREQUENCY)]
print("Number of days in the training data:", len(sampled_df_timeH_index))
sampled_df_timeH = sample_data(filtered_df, sampled_df_timeH_index, features)
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

tsv_combined = TimeSeriesVisualizer(test_df, features, 35, 35)
tsv_combined.update_sliders()
tsv_combined.show()
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
multi_window = WindowGenerator(input_width=INPUT_WIDTH, label_width=OUT_STEPS,   shift=OUT_STEPS,
                               train_df=train_df, val_df=val_df, test_df=test_df,
                               label_columns=features)
"""for feature in features:
    multi_window.plot(feature, 'Multi window')"""

multi_val_performance = dict()
multi_performance = dict()


def autoregressive_model():
    """
    # autoregressive RNN
    """
    feedback_model = FeedBack(32, OUT_STEPS, len(features), 20)
    prediction, state = feedback_model.warmup(multi_window.example[0])
    print(prediction.shape)
    print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)
    history = compile_and_fit(feedback_model, multi_window)

    IPython.display.clear_output()

    """multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val, return_dict=True)
    multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0, return_dict=True)
    for feature in features:
        multi_window.plot(feature, 'Autoregressive model predictions', feedback_model)"""
    return feedback_model


def multistep_cnn():
    multi_cnn = WideCNN(INPUT_WIDTH, OUT_STEPS, len(features), 20)
    IPython.display.clear_output()
    print('Output shape (batch, time, features): ', multi_cnn(multi_window.example[0]).shape)
    history = compile_and_fit(multi_cnn, multi_window)

    """multi_val_performance['CNN'] = multi_cnn.evaluate(multi_window.val, return_dict=True)
    multi_performance['CNN'] = multi_cnn.evaluate(multi_window.test, verbose=0, return_dict=True)
    for feature in features:
        multi_window.plot(feature, 'CNN model predictions', multi_cnn)"""
    return multi_cnn


def classification_datasets(features, feature_for_peaks):
    MIN_PEAK_HEIGHT = 0.3

    # Dataset is normalized
    train_df_peaks, _ = scipy.signal.find_peaks(train_df[feature_for_peaks], distance=10, height=MIN_PEAK_HEIGHT)
    val_df_peaks, _ = scipy.signal.find_peaks(val_df[feature_for_peaks], distance=10, height=MIN_PEAK_HEIGHT)
    test_df_peaks, _ = scipy.signal.find_peaks(test_df[feature_for_peaks], distance=10, height=MIN_PEAK_HEIGHT)

    train_inputs, train_labels = create_classification_dataset(train_df, features, train_df_peaks, INPUT_WIDTH, OUT_STEPS)
    val_inputs, val_labels = create_classification_dataset(val_df, features, val_df_peaks, INPUT_WIDTH, OUT_STEPS)
    test_inputs, test_labels = create_classification_dataset(test_df, features, test_df_peaks, INPUT_WIDTH, OUT_STEPS)
    return train_inputs, train_labels, val_inputs, val_labels


def classification_mlp(train_inputs, train_labels, val_inputs, val_labels, min_peak_distance=20):
    classification_model = ClassificationMLP(INPUT_WIDTH, OUT_STEPS, len(features), min_peak_distance)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      mode='min')
    classification_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                                 optimizer=tf.keras.optimizers.Adam(),
                                 metrics=[tf.keras.metrics.CategoricalCrossentropy()])
    history = classification_model.fit(x=train_inputs, y=train_labels, validation_data=(val_inputs, val_labels),
                             epochs=MAX_EPOCHS, callbacks=[early_stopping], shuffle=True, batch_size=32)
    return classification_model


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

period = sum(distances) / len(distances)
print("Period:", period)

sampled_test_df = test_df
#sampled_test_df, _ = normalize_df(sampled_test_df, method='own', values=norm_properties)
#sampled_test_df.index = (sampled_test_df.index - sampled_test_df.index[0]) / 24
#tf.config.run_functions_eagerly(True)
model_comparator = ModelComparator(sampled_test_df, INPUT_WIDTH, OUT_STEPS, features, features[0],
                                   plot=PLOT_TESTING, peak_comparison_distance=PEAK_COMPARISON_DISTANCE, step=1)
train_inputs, train_labels, val_inputs, val_labels = classification_datasets(features, features[0])
for run_id in range(NUM_RUNS):
    feedback_model = autoregressive_model()
    feedback_model._name = 'feed_back'
    multi_cnn_model = multistep_cnn()
    multi_cnn_model._name = 'wide_cnn'
    fitted_sin = NoisySinCurve(INPUT_WIDTH, OUT_STEPS, 1, train_df, features[0],
                               noise=0.0, period=period)
    fitted_sin._name = 'sin_curve'
    classification_model = classification_mlp(train_inputs, train_labels, val_inputs, val_labels, 24)
    classification_model._name = 'minPeakDist_24'
    models = [feedback_model, multi_cnn_model, fitted_sin, classification_model]
    #for i in range(2, 37, 3):
    #    model = classification_mlp(train_inputs, train_labels, val_inputs, val_labels, i)
    #    model._name = 'minPeakDist_' + str(i)
    #    models.append(model)
    #combined = mmml(feedback_model, multi_cnn_model)
    #combined._name = 'combined_RNN_CNN'
    saved_models_paths = []
    if SAVE_MODELS:
        for model in models:
            model_name = model._name + "_RUN" + str(run_id) + "_IN" + str(INPUT_WIDTH)
            model_save_path_full = os.path.join(save_models_dir, model_name)
            saved_models_paths.append(model_save_path_full)
            model.save(model_save_path_full)
    list_of_models = models  # []
    """for model_name in saved_models_paths:
        model = tf.keras.models.load_model(model_name,
                custom_objects={'FeedBack': FeedBack, 'WideCNN': WideCNN,
                                'ClassificationMLP': ClassificationMLP, 'Peak_loss': Peak_loss})
        list_of_models.append(model)"""
    model_comparator.compare_models(list_of_models, run_id)
    model_comparator.plot_pred_peak_distribution(run_id)
    tsv = TimeSeriesVisualizer(sampled_test_df, features, INPUT_WIDTH, OUT_STEPS)
    tsv.update_sliders(list_of_models)
    tsv.show()

#multistep_performance()
model_comparator.print_peak_statistics()
model_comparator.plot_in_out_peaks()

