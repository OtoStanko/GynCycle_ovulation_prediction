import os

import numpy as np
import pandas as pd
import tensorflow as tf


def create_dataframe(input_files_directory, features, time_file_prefix, feature_file_suffix='1'):
    """
    Creates a pandas dataframe from csv files containing time series data.
    The function assumes existence of the directory where the files are stored.
    The time file name is expected to be in form {time_file_prefix}_{feature_file_suffix}.csv i.e.(Time_1.csv)
    The feature files' names are expected to be in form {feature}_{feature_file_suffix}.csv,
    where feature is a value from the features' parameter.

    So far, nothing is checked. If any error rises due to any reason (files not found...) function fails.

    Function doesn't set the time as an index

    :param input_files_directory: path to the directory with time and features files
    :param features: list of strings where each string in a prefix of the corresponding feature file
    :param time_file_prefix: prefix of the file containing the timestamps of the time series.
    :param feature_file_suffix:
    :return: dataframe with timestamp and the features' tracks
    """
    time_file = os.path.join(input_files_directory, "{}_{}.csv".format(time_file_prefix, feature_file_suffix))
    times = pd.read_csv(time_file, header=None, names=[time_file_prefix])
    hormone_levels = [times]
    for feature in features:
        feature_file = os.path.join(input_files_directory, "{}_{}.csv".format(feature, feature_file_suffix))
        feature_values = pd.read_csv(feature_file, header=None, names=[feature])
        hormone_levels.append(feature_values)
    combined_df = pd.concat(hormone_levels, axis=1)
    return combined_df


def sample_data(original_df, new_index, features):
    """
    For a given dataframe and an index returns a new dataframe with values at the time points from the new index
    sampled from the original dataframe using linear interpolation.
    :param original_df: pandas dataframe with time set as index. Time of type integer is expected, no datetime
    :param new_index: an iterable with time stamps at which new samples should be made
    :param features: list of features from the original df that should be sampled
    :return: new df with sampled values at the time points from the new index. The new index is set as an index
    """
    """
        for every time in the new index, find the largest smaller value and smallest larger value
        and interpolate them to get the new value
        Edge case if at least one of the is the same time
    """
    hormone_levels = {key: [] for key in features}
    i = 0
    for curr_time in new_index:
        while original_df.index[i + 1] < curr_time:
            i += 1
        for feature in features:
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
    for feature in features:
        sampled_df[feature] = np.array(hormone_levels[feature])
    sampled_df.index = new_index
    sampled_df.index.name = 'DateTime'
    return sampled_df


def normalize_df(df, method='standard', values=None):
    """
    :param df: dataframe to normalize
    :param method:  'standard' for standardization,
                    'minmax' for minmax normalization,
                    'own' for 'standardization' with given values
    :param values: a dict of form {feature, (a, b)}, where feature is a feature from the df and a,b depends on the method.
    for standardization, if specified, a is mean and b is std to be used. If not specified, both will be calculated
    for each feature separately. For minmax, a is the lower bound of the interval and b is the upper bound of the interval.
    If not specified, a=0, b=1. For own method, a and b must be specified. The behaviour is technically the same as in the standardization.
    a is subtracted and the result is divided by b. But the a and b doesn't have to be mean and std, so to avoid confusion of author and others,
    we have included it as a separate method of normalization.
    :return: normalized df
    """
    """
    methods: standardization, mean and std may be provided, otherwise are calculated values=(mean, std) is expected
             minmax, if no values are provided, the scale to [0, 1] is done, otherwise to [a, b]
    """
    df = df.copy()
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


def create_classification_dataset(df, features, peaks, input_window_length, output_length):
    """
    Creates dataset of inputs and labels for classification of peaks in the output from multiple input features.
    :param df: dataframe containing time series data
    :param features: list of features from the df that should be incorporated in the classification dataset
    :param peaks: list of indexes at which peaks are present in the df -> df.index(peaks) should return the times
     at which the peaks are present
    :param input_window_length: length of the input
    :param output_length: length of the output (label)
    :return: tuple (dims=2) containing input data of shape (len(df.index)-input_window_length-output_length+1, input_window_length, len(features))
    and labels data of shape (len(df.index)-input_window_length-output_length+1, output_length, 1)
    """
    inputs = []
    labels = []

    # Create the training dataset
    for start_time in range(len(df.index)-input_window_length):
        end_time = start_time + input_window_length

        # Get input features for the 7-day window
        input_data = []
        """for i in range(start_time, end_time):
            feature_inputs = []
            for feature in features:
                feature_inputs.append(df[feature].values[i])
            input_data.append(feature_inputs)"""
        for feature in features:
            feature_inputs = df[feature][start_time:end_time].values
            input_data.append(feature_inputs)
        input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Find the next peak time after the end of the input window
        next_peaks = peaks[peaks >= end_time]

        if len(next_peaks) != 0:
            # Calculate the label as the time difference in seconds
            inputs.append(input_data)
            label = (next_peaks[0] - end_time)
            label_vector = [0 for _ in range(output_length)]
            if label < output_length:
                label_vector[label] = 1
            labels.append(tf.convert_to_tensor([label_vector], dtype=tf.int32))
        else:
            break
    inputs = tf.stack(inputs)
    labels = tf.stack(labels)

    print("Input shape:", inputs.shape)
    print("Labels shape:", labels.shape)
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