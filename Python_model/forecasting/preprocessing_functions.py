import os

import tensorflow as tf
import numpy as np
import pandas as pd


def create_dataframe(input_files_directory, features, time_file_prefix, feature_file_suffix='1'):
    time_file = os.path.join(input_files_directory, "{}_{}.csv".format(time_file_prefix, feature_file_suffix))
    times = pd.read_csv(time_file, header=None, names=[time_file_prefix])
    hormone_levels = [times]
    for feature in features:
        feature_file = os.path.join(input_files_directory, "{}_{}.csv".format(feature, feature_file_suffix))
        feature_values = pd.read_csv(feature_file, header=None, names=[feature])
        hormone_levels.append(feature_values)
    combined_df = pd.concat(hormone_levels, axis=1)
    return combined_df


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
            x0 = original_df.index[i]
            y0 = original_df[feature][original_df.index[i]]
            x1 = original_df.index[i + 1]
            y1 = original_df[feature][original_df.index[i + 1]]
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


def create_classification_dataset(df, feature, peaks, input_window_length):
    inputs = []
    labels = []

    # Create the training dataset
    for start_time in range(len(df.index)-input_window_length):
        end_time = start_time + input_window_length

        # Get input features for the 7-day window
        input_data = tf.convert_to_tensor(df[feature][start_time:end_time].values, dtype=tf.float32)

        # Find the next peak time after the end of the input window
        next_peaks = peaks[peaks >= end_time]
        #next_peaks = next_peaks[next_peaks < end_time + 35]

        if len(next_peaks) != 0:
            # Calculate the label as the time difference in seconds
            #inputs.append([input_data])
            inputs.append(tf.expand_dims(input_data, axis=0))
            label = (next_peaks[0] - end_time)
            label_vector = [0 for _ in range(35)]
            if label < 35:
                label_vector[label] = 1
            #labels.append([label_vector])
            labels.append(tf.convert_to_tensor([label_vector], dtype=tf.int32))
        else:
            break

    # Convert to NumPy arrays
    inputs = tf.stack(inputs)
    labels = tf.stack(labels)

    print("Input shape:", inputs.shape)
    print("Labels shape:", labels.shape)
    print("Peaks:", peaks)
    print("DF length:", len(df.index))
    print("Num records:", len(labels))
    return inputs, labels


def create_KAN_dataset(df, feature):
    inputs = []
    labels = []
    for x in range(len(df.index)):
        inputs.append([x])
        labels.append([df[feature][df.index[x]]])
    return np.array(inputs), np.array(labels)