import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX


def sample_data(original_df, new_index, column):
    # The records are not evenly distributed. We will do sampling with linear interpolation for the models
    """
        for every time in the new index, find the largest smaller value and smallest larger value
        and interpolate them to get the new value
        Edge case if at least one of the is the same time
    """
    FSH_levels = np.array([])
    i = 0
    for curr_time in new_index:
        while original_df.index[i + 1] < curr_time:
            i += 1
        # index_of_largest_smaller_time = i
        x0 = largest_smaller_time = original_df.index[i]
        y0 = largest_smaller_value = original_df[column][original_df.index[i]]
        x1 = smallest_larger_time = original_df.index[i + 1]
        y1 = smallest_larger_value = original_df[column][original_df.index[i + 1]]
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
        FSH_levels = np.append(FSH_levels, y)

    sampled_df = pd.DataFrame()
    sampled_df[column] = FSH_levels
    sampled_df.index = new_index
    sampled_df.index.name = 'DateTime'
    return sampled_df


# Set the parameters
workDir = os.path.join(os.getcwd(), "../outputDir/")
sampling_frequency = 1
sampling_frequency_unit = 'H'
num_initial_days_to_discard = 50
train_test_split_days = 250
test_days_end = 300
hormone = 'FSH'


timeFile = os.path.join(workDir, "Time_1.csv")
FSHFile = os.path.join(workDir, "{}_1.csv".format(hormone))
times = pd.read_csv(timeFile, header=None, names=['Time'])
fsh_levels = pd.read_csv(FSHFile, header=None, names=[hormone])

combined_df = pd.concat([times, fsh_levels], axis=1)
combined_df['Time'] = combined_df['Time'] * 24


# Plot the loaded data
sns.set()
plt.ylabel('{} level'.format(hormone))
plt.xlabel('Time in days')
plt.xticks(rotation=45)
plt.plot(combined_df['Time'], combined_df[hormone], )
plt.show()


# Set some starting date that will be needed for index
start_date = pd.Timestamp('2024-01-01')
# We will not use first 50 days of the simulation
data_start_date = start_date + pd.to_timedelta(num_initial_days_to_discard*24, 'h')
# Set the split date for the train and test data
data_tt_split_date = start_date + pd.to_timedelta(train_test_split_days*24, 'h')
# Set the stop date for the test data (end day of the data)
data_stop_date = start_date + pd.to_timedelta(test_days_end * 24, 'h')

# First 50 days of the simulation may be a bit messy and thus we ignore them
filtered_df = combined_df[combined_df['Time'] > num_initial_days_to_discard*24]
filtered_df.set_index('Time', inplace=True)

print(filtered_df.describe().transpose())
filtered_df_timeH = filtered_df.copy()
print(filtered_df[hormone])
print(filtered_df_timeH[hormone])


time_delta = pd.to_timedelta(filtered_df.index, unit='h')
datetime_index = start_date + time_delta
filtered_df.index = datetime_index
filtered_df.index.name = 'DateTime'
print(filtered_df[hormone][0])


# The records are not evenly distributed. We will do sampling with linear interpolation for the models
new_index = pd.date_range(start=data_start_date,
                          end=data_stop_date,
                          freq="{}{}".format(sampling_frequency, sampling_frequency_unit))

sampled_df = sample_data(filtered_df, new_index, hormone)

plt.plot(sampled_df.index, sampled_df[hormone], )
plt.show()



sapmled_df_timeH_index = [i for i in range(num_initial_days_to_discard*24, test_days_end*24+1)]
print(sapmled_df_timeH_index)
print(len(sapmled_df_timeH_index))
sampled_df_timeH = sample_data(filtered_df_timeH, sapmled_df_timeH_index, hormone)

plt.plot(sampled_df_timeH.index, sampled_df_timeH[hormone], )
plt.show()

sampled_df_timeH['Month_sin'] = (np.sin((sampled_df_timeH.index * ((2 * np.pi / (25*24))))) * 5) + 15
plt.plot(sampled_df_timeH['Month_sin'])
plt.plot(sampled_df_timeH[hormone])
plt.show()

fft = tf.signal.rfft(sampled_df_timeH[hormone])
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
plt.show()

column_indices = {name: i for i, name in enumerate(sampled_df_timeH.columns)}

n = len(sampled_df_timeH)
train_df = sampled_df_timeH[0:int(n*0.7)]
val_df = sampled_df_timeH[int(n*0.7):int(n*0.9)]
test_df = sampled_df_timeH[int(n*0.9):]

num_features = sampled_df_timeH.shape[1]
print(num_features)

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

plt.plot(train_df.index, train_df[hormone], color='black')
plt.plot(val_df.index, val_df[hormone], color='blue')
plt.plot(test_df.index, test_df[hormone], color='red')
plt.show()






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
