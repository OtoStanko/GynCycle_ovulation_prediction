import scipy.signal
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from preprocessing_functions import *

def plot_dataset(dataset_to_plot, labels_to_plot):
    for i in range(len(dataset_to_plot)):
        if i+imlp < len(dataset_to_plot) and i % 5 == 0:
            plt.plot(np.linspace(0, imlp, imlp), dataset_to_plot[i], color='blue')
            plt.plot(np.linspace(imlp, imlp+35, 35), dataset_to_plot[i+imlp], color='blue')
            plt.plot(np.linspace(imlp, imlp+35, 35), labels_to_plot[i], label='test_label')
            plt.axvline(x=imlp, color='r', linestyle='--', )
            plt.legend()
            plt.show()


TRAIN_DATA_SUFFIX = 'of_4'
TEST_DATA_SUFFIX = 'of_1'

# Set the parameters
workDir = os.path.join(os.getcwd(), "../outputDir/")
SAMPLING_FREQUENCY = 24
SAMPLING_FREQUENCY_UNIT = 'H'
NUM_INITIAL_DAYS_TO_DISCARD = 50
features = ['LH']
MAX_EPOCHS = 25

OUT_STEPS = 35
INPUT_WIDTH = 35
MIN_PEAK_HEIGHT = 20

combined_df = create_dataframe(workDir, features, 'Time', TRAIN_DATA_SUFFIX)
combined_df['Time'] = combined_df['Time'] * 24

filtered_df = combined_df[combined_df['Time'] > NUM_INITIAL_DAYS_TO_DISCARD * 24]
filtered_df.set_index('Time', inplace=True)

sampled_df_timeH_index = [i for i in range(NUM_INITIAL_DAYS_TO_DISCARD * 24, int(filtered_df.index[-1]) + 1, SAMPLING_FREQUENCY)]
print("Number of days in the training data:", len(sampled_df_timeH_index))
sampled_df_timeH = sample_data(filtered_df, sampled_df_timeH_index, features)


n = len(sampled_df_timeH)
train_df = sampled_df_timeH[0:int(n*0.7)]
val_df = sampled_df_timeH[int(n*0.7):int(n*0.9)]
test_df = sampled_df_timeH[int(n*0.9):]

print("# days in training part:", len(train_df))
print("# days testing part:", len(test_df))



train_df_peaks, _ = scipy.signal.find_peaks(train_df[features[0]], distance=10, height=MIN_PEAK_HEIGHT)
val_df_peaks, _ = scipy.signal.find_peaks(val_df[features[0]], distance=10, height=MIN_PEAK_HEIGHT)
test_df_peaks, _ = scipy.signal.find_peaks(test_df[features[0]], distance=10, height=MIN_PEAK_HEIGHT)
imlp = 35
train_inputs, train_labels = create_classification_dataset(train_df, features[0], train_df_peaks, imlp)
val_inputs, val_labels = create_classification_dataset(val_df, features[0], val_df_peaks, imlp)
test_inputs, test_labels = create_classification_dataset(test_df, features[0], test_df_peaks, imlp)
print(len(train_inputs), len(val_inputs), len(test_inputs))

mean = np.mean(train_inputs, axis=0)
train_max = np.max(train_inputs, axis=0)
std = np.std(train_inputs, axis=0)
train_df_inputs_norm = train_inputs / train_max  # (train_inputs - mean) / std
val_df_inputs_norm = val_inputs / train_max  # (val_inputs - mean) / std
test_df_inputs_norm = test_inputs / train_max  # (test_inputs - mean) / std

#plot_dataset(train_df_inputs_norm, train_labels)


simple_mlp = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=35, activation='sigmoid')
])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    mode='min')
simple_mlp.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.CategoricalCrossentropy()])
history = simple_mlp.fit(x=train_df_inputs_norm, y=train_labels, validation_data=(val_df_inputs_norm, val_labels),
                         epochs=MAX_EPOCHS, callbacks=[early_stopping], shuffle=True)
simpl_mlp_results = simple_mlp.evaluate(x=test_df_inputs_norm, y=test_labels)

predictions = []
for input_val in test_df_inputs_norm:
    inputs = tf.convert_to_tensor(input_val, dtype=tf.float32)
    inputs = tf.reshape(inputs, (1, 1, imlp))
    prediction = simple_mlp(inputs)
    predictions.append(prediction[0])

results = []
print("Num predictions:", len(predictions))
for i in range(len(predictions)):
    prediction = predictions[i]
    prediction = tf.reshape(prediction, (imlp))
    prediction = prediction / 3
    smoothened_prediction = savgol_filter(prediction, 11, 2)
    max_index = np.unravel_index(np.argmax(prediction, axis=None), prediction.shape)
    result = np.zeros(len(prediction), dtype=int)
    result[max_index] = int(1)
    results.append(result)
    if i+imlp < len(predictions) and i % 5 == 0:
        plt.plot(np.linspace(0, imlp, imlp), test_df_inputs_norm[i][0], color='blue')
        plt.plot(np.linspace(imlp, imlp+35, 35), test_df_inputs_norm[i+imlp][0], color='blue')
        plt.plot(np.linspace(imlp, imlp+35, 35), test_labels[i][0], label='test_label')
        plt.plot(np.linspace(imlp, imlp+35, 35), prediction, label='prediction sigmoid', color='green')
        plt.plot(np.linspace(imlp, imlp+35, 35), smoothened_prediction, label='smoothened prediction', color='red')
        plt.plot(np.linspace(imlp, imlp + 35, 35), result, label='prediction peak', color='red')
        plt.axvline(x=imlp, color='r', linestyle='--', )
        plt.legend(loc='upper left')
        plt.show()
