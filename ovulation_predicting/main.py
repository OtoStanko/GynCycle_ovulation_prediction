from collections import Counter
from tensorflow.keras.callbacks import TensorBoard

import matplotlib.pyplot as plt
import scipy.signal
import seaborn as sns

from ModelComparator import ModelComparator
from ovulation_predicting.models import MyModelWrapper, NoisySinCurve
from preprocessing_functions import *
from supporting_scripts import print_ts
from TimeSeriesVisualizer import TimeSeriesVisualizer
from windowGenerator import WindowGenerator


"""
    Parameters
"""
#
INPUT_DIR = os.path.join(os.getcwd(), "../Python_model/outputDir/")
TRAIN_DATA_SUFFIX = '1_n'
TEST_DATA_SUFFIX = 'of_1'
SAVE_MODELS_DIR = os.path.join(os.getcwd(), "./saved_models/")

LOSS_FUNCTIONS = [tf.keras.losses.MeanSquaredError()]

SAMPLING_FREQUENCY = 24
SAMPLING_FREQUENCY_UNIT = 'H'
NUM_INITIAL_DAYS_TO_DISCARD = 50
features = ['LH', 'E2']
MAX_EPOCHS = 25

# forecast parameters
INPUT_WIDTH = 35
OUT_STEPS = 35

NUM_RUNS = 1
PEAK_COMPARISON_DISTANCE = 2
PLOT_TESTING = False
SAVE_MODELS = False


# test on a small TS
test_dataframe = create_dataframe(INPUT_DIR, features, 'Time', TEST_DATA_SUFFIX)
test_dataframe['Time'] = test_dataframe['Time'] * 24
# train on a long TS
combined_df = create_dataframe(INPUT_DIR, features, 'Time', TRAIN_DATA_SUFFIX)
combined_df['Time'] = combined_df['Time'] * 24
print('Number of records in the loaded data for training:', len(combined_df['Time']))

# Plot the loaded data
sns.set()
print_ts(combined_df['Time'], combined_df[features],
        'Time in hours', '{} levels'.format('Hormones'),
         'Raw dataset {}'.format(features))

# The first 50 days of the simulation may be a bit messy and thus we ignore them
filtered_test_df = test_dataframe[test_dataframe['Time'] > NUM_INITIAL_DAYS_TO_DISCARD * 24]
filtered_test_df.set_index('Time', inplace=True)
filtered_df = combined_df[combined_df['Time'] > NUM_INITIAL_DAYS_TO_DISCARD * 24]
filtered_df.set_index('Time', inplace=True)


# Sample the time series
index_for_ts_sampling = [i for i in range(NUM_INITIAL_DAYS_TO_DISCARD * 24, int(filtered_df.index[-1]) + 1, SAMPLING_FREQUENCY)]
print("Number of days in the training data:", len(index_for_ts_sampling))
sampled_ts = sample_data(filtered_df, index_for_ts_sampling, features)
print('Num records in the sampled dataframe with raw hours: '
      '(Should be the same as the number of days in the training data)', len(sampled_ts.index))
print_ts(sampled_ts.index, sampled_ts[features],
        'Time in hours', '{} levels'.format('Hormones'),
         'Sampled dataframe with raw hours')

index_for_test_ts_sampling = [i for i in range(NUM_INITIAL_DAYS_TO_DISCARD * 24, int(filtered_test_df.index[-1]) + 1, SAMPLING_FREQUENCY)]
sampled_test_ts = sample_data(filtered_test_df, index_for_test_ts_sampling, features)
print("Number of days in the testing data:", len(index_for_test_ts_sampling))


column_indices = {name: i for i, name in enumerate(sampled_ts.columns)}
n = len(sampled_ts)
train_df = sampled_ts[0:int(n * 0.7)]
val_df = sampled_ts[int(n * 0.7):int(n * 0.9)]
test_df = sampled_ts[int(n * 0.9):]

num_features = sampled_ts.shape[1]
print("Number of features sanity check", num_features, len(features))

train_mean = train_df.mean()
train_std = train_df.std()


train_df, norm_properties = normalize_df(train_df, method='minmax', values={feature: (0, 1) for feature in features})
val_df, _ = normalize_df(val_df, method='own', values=norm_properties)
test_df, _ = normalize_df(test_df, method='own', values=norm_properties)

#train_df, _ = normalize_df(train_df, method='log')
#val_df, _ = normalize_df(val_df, method='log')
#test_df, _ = normalize_df(test_df, method='log')


for feature in features:
    plt.plot(train_df.index, train_df[feature], color='#1f77b4')
    plt.plot(val_df.index, val_df[feature], color='blue')
    plt.plot(test_df.index, test_df[feature], color='chocolate')
    plt.title("Preprocessed split dataset ['{}']".format(feature))
    plt.ylabel('Normalized hormone levels')
    plt.xlabel('Time in hours')
    plt.show()

tsv_combined = TimeSeriesVisualizer(test_df, features, 35, 35)
tsv_combined.update_sliders()
tsv_combined.show()


"""
# Multi-step models
"""
multi_window = WindowGenerator(input_width=INPUT_WIDTH, label_width=OUT_STEPS,   shift=OUT_STEPS,
                               train_df=train_df, val_df=val_df, test_df=test_df,
                               label_columns=features)


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

sampled_test_ts = test_df
#sampled_test_df, _ = normalize_df(sampled_test_df, method='own', values=norm_properties)
##sampled_test_df.index = (sampled_test_df.index - sampled_test_df.index[0]) / 24
#tf.config.run_functions_eagerly(True)
model_comparator = ModelComparator(sampled_test_ts, INPUT_WIDTH, OUT_STEPS, features, features[0],
                                   plot=PLOT_TESTING, peak_comparison_distance=PEAK_COMPARISON_DISTANCE, step=1)
model_wrapper = MyModelWrapper(features, INPUT_WIDTH, OUT_STEPS, multi_window, LOSS_FUNCTIONS, MAX_EPOCHS)

train_inputs, train_labels, val_inputs, val_labels = model_wrapper.classification_datasets(
    train_df, val_df, test_df, [features[0]], features[0])
for run_id in range(NUM_RUNS):
    feedback_model = model_wrapper.autoregressive_model()
    feedback_model._name = 'RNN'
    multi_cnn_model = model_wrapper.multistep_cnn()
    multi_cnn_model._name = 'CNN'
    fitted_sin = NoisySinCurve(INPUT_WIDTH, OUT_STEPS, 1, train_df, features[0],
                               noise=0.0, period=period)
    fitted_sin._name = 'Baseline'
    cnn_lstm_model = model_wrapper.cnn_lstm(filters=[256, 128, 64], ks=[4, 3, 2], dilations=[1, 2, 4])
    cnn_lstm_model._name = 'CNN+LSTM'
    #classification_model = classification_mlp(train_inputs, train_labels, val_inputs, val_labels, 24)
    #classification_model._name = 'Classifier'
    models = [feedback_model, multi_cnn_model, fitted_sin, cnn_lstm_model]
    saved_models_paths = []
    if SAVE_MODELS:
        for model in models:
            model_name = model._name + "_RUN" + str(run_id) + "_IN" + str(INPUT_WIDTH)
            model_save_path_full = os.path.join(SAVE_MODELS_DIR, model_name)
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
    #sampled_test_df.to_csv(f"{inputDir}atsv_df.csv")
    """for column in sampled_test_df.columns:
        sampled_test_df[[column]].to_csv(f"{inputDir}atsv_{column}.csv", index=False, header=False)
    df_index_data = np.array(sampled_test_df.index) - sampled_test_df.index[0]
    np.savetxt("../outputDir/atsv_time.csv", df_index_data, delimiter="\t", fmt='%d')"""
    tsv = TimeSeriesVisualizer(sampled_test_ts, features, INPUT_WIDTH, OUT_STEPS)
    tsv.update_sliders(list_of_models)
    tsv.show()

model_comparator.print_peak_statistics()
model_comparator.plot_in_out_peaks()

