import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from ovulation_predicting.preprocessing_functions import create_dataframe, sample_data
from ovulation_predicting.supporting_scripts import print_ts, show_plot


def plot_temp(df):
    #plt.figure(figsize=(10, 6))
    for subject_id, row in df.iterrows():
        plt.plot(row.index, row.values, marker='o', label=f"Subject {subject_id}")
    plt.grid(True, alpha=0.3)
    show_plot()


SAMPLING_FREQUENCY = 1
TRAIN_DATA_SUFFIX = "1"

features = ["LH","P4",]
INPUT_DIR = os.path.join(os.getcwd(), "..", "MATLAB_model", "hormone_populations")

combined_df = create_dataframe(INPUT_DIR, features, 'Time', TRAIN_DATA_SUFFIX)
combined_df.set_index('Time', inplace=True)

index_for_ts_sampling = [i for i in range(0, int(combined_df.index[-1])+1, SAMPLING_FREQUENCY)]
print("Number of days in the training data:", len(index_for_ts_sampling))
sampled_ts = sample_data(combined_df, index_for_ts_sampling, features)

p4_peaks = find_peaks(sampled_ts["P4"])
print(p4_peaks)
print_ts(sampled_ts.index, sampled_ts[features],
        'Time in hours', '{} levels'.format('Hormones'),
         'Sampled dataframe with raw hours')

temps = pd.read_csv(os.path.join(INPUT_DIR, "simulated_bbt_30days.csv"), parse_dates=["date"])
df_subject_wide = temps.pivot(index="subject_id", columns="day_of_cycle", values="bbt_C")
df_subject_wide = df_subject_wide.reindex(sorted(df_subject_wide.columns), axis=1)
print(df_subject_wide)
plot_temp(df_subject_wide)

rise_threshold = 0.2
lookback = 3
aligned_data = {}

for subject_id, temps in df_subject_wide.iterrows():
    temps = temps.values
    ovulation_day = None
    for d in range(lookback, len(temps)):
        baseline = np.mean(temps[d - lookback:d])
        if temps[d] - baseline >= rise_threshold:
            ovulation_day = d
            break

    if ovulation_day is None:
        ovulation_day = np.argmax(temps)

    shifted_days = np.arange(len(temps)) - ovulation_day
    aligned_data[subject_id] = dict(zip(shifted_days, temps))
df_aligned = pd.DataFrame.from_dict(aligned_data, orient="index")
plot_temp(df_aligned)

common_start = df_aligned.columns.min()
common_end = df_aligned.columns.max()
for subj in df_aligned.index:
    subj_days = df_aligned.loc[subj].dropna().index
    common_start = max(common_start, subj_days.min())
    common_end = min(common_end, subj_days.max())
df_aligned_trimmed = df_aligned.loc[:, common_start:common_end]
plot_temp(df_aligned_trimmed)

p4_series = sampled_ts["P4"]
p4_anchor_day = p4_peaks[0][1]
print(p4_anchor_day)
df_aligned_to_p4 = df_aligned_trimmed.copy()
df_aligned_to_p4.columns = df_aligned_to_p4.columns + p4_anchor_day
common_days = df_aligned_to_p4.columns.intersection(p4_series.index)
p4_aligned = p4_series.loc[common_days]
bbt_aligned = df_aligned_to_p4[common_days]

common_start = max(bbt_aligned.columns.min(), p4_aligned.index.min())
common_end   = min(bbt_aligned.columns.max(), p4_aligned.index.max())
bbt_window = bbt_aligned.loc[:, common_start:common_end]
p4_window  = p4_aligned.loc[common_start:common_end]

print("Trimmed BBT:")
print(bbt_window.head())
print("\nTrimmed P4:")
print(p4_window.head())

days = bbt_window.columns.astype(int)

fig, ax1 = plt.subplots(figsize=(12,6))

# Plot BBT for each subject
for subject in bbt_window.index:
    ax1.plot(days, bbt_window.loc[subject], marker='o', label=f'{subject} BBT')

ax1.set_xlabel('Day')
ax1.set_ylabel('BBT (°C)')
ax1.tick_params(axis='y')

# Create a second y-axis for P4
ax2 = ax1.twinx()
ax2.plot(days, p4_window, color='red', marker='x', linewidth=2, label='P4')
ax2.set_ylabel('P4 (ng/mL)')
ax2.tick_params(axis='y', labelcolor='red')

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')
plt.title('BBT and P4 over Time')
show_plot()

temperature_prev = []
temperature_new = []
p4_values = []

for subject in bbt_window.index:
    temps = bbt_window.loc[subject].values
    for i in range(1, len(temps)):
        temperature_prev.append(temps[i-1])
        temperature_new.append(temps[i])
        p4_values.append(p4_window.iloc[i])  # assuming day alignment
temperature_prev = np.array(temperature_prev)
temperature_new = np.array(temperature_new)
p4_values = np.array(p4_values)

def model_combined(X, a, b, c):
    prev_temp, p4 = X
    return prev_temp + a * ((b + c * p4) - prev_temp)

popt, pcov = curve_fit(
    model_combined,
    (temperature_prev, p4_values),
    temperature_new,
    p0=[0.5, 35.5, 1],
)

a_fit, b_fit, c_fit = popt
print("Fitted parameters:", a_fit, b_fit, c_fit)
