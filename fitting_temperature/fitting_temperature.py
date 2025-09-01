import os

from ovulation_predicting.preprocessing_functions import create_dataframe, sample_data
from ovulation_predicting.supporting_scripts import print_ts


SAMPLING_FREQUENCY = 1
TRAIN_DATA_SUFFIX = "1"

features = ["LH","P4",]
INPUT_DIR = os.path.join(os.getcwd(), "..", "MATLAB_model", "hormone_populations")

combined_df = create_dataframe(INPUT_DIR, features, 'Time', TRAIN_DATA_SUFFIX)
combined_df.set_index('Time', inplace=True)

index_for_ts_sampling = [i for i in range(0, int(combined_df.index[-1])+1, SAMPLING_FREQUENCY)]
print("Number of days in the training data:", len(index_for_ts_sampling))
sampled_ts = sample_data(combined_df, index_for_ts_sampling, features)

print_ts(sampled_ts.index, sampled_ts[features],
        'Time in hours', '{} levels'.format('Hormones'),
         'Sampled dataframe with raw hours')
