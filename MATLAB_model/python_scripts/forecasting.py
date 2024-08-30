import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set the parameters
workDir = os.path.join(os.getcwd(), "../outputDir/")
sampling_frequency = 1
sampling_frequency_unit = 'H'
num_initial_days_to_discard = 50
train_test_split_days = 250
Test_days_end = 300
hormone = 'LH'


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
data_stop_date = start_date + pd.to_timedelta(Test_days_end*24, 'h')

# First 50 days of the simulation may be a bit messy and thus we ignore them
filtered_df = combined_df[combined_df['Time'] > num_initial_days_to_discard*24]
filtered_df.set_index('Time', inplace=True)

time_delta = pd.to_timedelta(filtered_df.index, unit='h')
datetime_index = start_date + time_delta
filtered_df.index = datetime_index
filtered_df.index.name = 'DateTime'

# The records are not evenly distributed. We will do sampling with linear interpolation for the models
new_index = pd.date_range(start=data_start_date,
                          end=data_stop_date,
                          freq="{}{}".format(sampling_frequency, sampling_frequency_unit))

"""
    for every time in the new index, find the largest smaller value and smallest larger value
    and interpolate them to get the new value
    Edge case if at least one of the is the same time
"""
FSH_levels = np.array([])
i = 0
for curr_time in new_index:
    while filtered_df.index[i+1] < curr_time:
        i += 1
    #index_of_largest_smaller_time = i
    x0 = largest_smaller_time = filtered_df.index[i]
    y0 = largest_smaller_value = filtered_df[hormone][i]
    x1 = smallest_larger_time = filtered_df.index[i+1]
    y1 = smallest_larger_value = filtered_df[hormone][i+1]
    x = curr_time
    diff_x1_x = (x1 - x).total_seconds()
    diff_x_x0 = (x - x0).total_seconds()
    diff_x1_x0 = (x1 - x0).total_seconds()
    y = y0 * (diff_x1_x/diff_x1_x0) + y1 * (diff_x_x0/diff_x1_x0)
    FSH_levels = np.append(FSH_levels, y)

sampled_df = pd.DataFrame()
sampled_df[hormone] = FSH_levels
sampled_df.index = new_index
sampled_df.index.name = 'DateTime'

# We will do some comparison of the original data and the sampled data

# Split the data into training and testing sets
train_df = filtered_df[(filtered_df.index > data_start_date) & (filtered_df.index <= data_tt_split_date)]
test_df = filtered_df[(filtered_df.index > data_tt_split_date) & (filtered_df.index <= data_stop_date)]

plt.plot(train_df.index, train_df[hormone], color = "black")
plt.plot(test_df.index, test_df[hormone], color = "red")
plt.show()

sampled_train_df = sampled_df[(sampled_df.index > data_start_date) & (sampled_df.index <= data_tt_split_date)]
sampled_test_df = sampled_df[(sampled_df.index > data_tt_split_date) & (sampled_df.index <= data_stop_date)]

plt.plot(sampled_train_df.index, sampled_train_df[hormone], color = "black")
plt.plot(sampled_test_df.index, sampled_test_df[hormone], color = "red")
plt.show()

# Fit the model
mod = SARIMAX(sampled_df[hormone], trend='c', order=(2,1,0), seasonal_order=(1,1,0,12),
              freq="{}{}".format(sampling_frequency, sampling_frequency_unit))
res = mod.fit(disp=False)
print(res.summary())
