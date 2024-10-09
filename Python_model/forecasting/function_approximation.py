from kan import *
import numpy as np
import torch
import matplotlib.pyplot as plt

from preprocessing_functions import create_dataframe, sample_data, create_KAN_dataset

TRAIN_DATA_SUFFIX = 4
workDir = os.path.join(os.getcwd(), "../outputDir/")
SAMPLING_FREQUENCY = 24
SAMPLING_FREQUENCY_UNIT = 'H'
NUM_INITIAL_DAYS_TO_DISCARD = 50
features = ['LH']


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

train_inputs, train_labels = create_KAN_dataset(train_df, 'LH')
val_inputs, val_labels = create_KAN_dataset(val_df, 'LH')
test_inputs, test_labels = create_KAN_dataset(test_df, 'LH')

datasets = []

n_peak = 5
n_num_per_peak = 100
n_sample = n_peak * n_num_per_peak

x_grid = torch.linspace(-1,1,steps=n_sample)

x_centers = 2/n_peak * (np.arange(n_peak) - n_peak/2+0.5)

x_sample = torch.stack([torch.linspace(-1/n_peak,1/n_peak,steps=n_num_per_peak)+center for center in x_centers]).reshape(-1,)


y = 0.
for center in x_centers:
    y += torch.exp(-(x_grid-center)**2*300)

y_sample = 0.
for center in x_centers:
    y_sample += torch.exp(-(x_sample-center)**2*300)


plt.plot(x_grid.detach().numpy(), y.detach().numpy())
plt.scatter(x_sample.detach().numpy(), y_sample.detach().numpy())
plt.show()

plt.subplots(1, 5, figsize=(15, 2))
plt.subplots_adjust(wspace=0, hspace=0)

for i in range(1,6):
    plt.subplot(1,5,i)
    group_id = i - 1
    plt.plot(x_grid.detach().numpy(), y.detach().numpy(), color='black', alpha=0.1)
    plt.scatter(x_sample[group_id*n_num_per_peak:(group_id+1)*n_num_per_peak].detach().numpy(),
                y_sample[group_id*n_num_per_peak:(group_id+1)*n_num_per_peak].detach().numpy(),
                color="black", s=2)
    plt.xlim(-1,1)
    plt.ylim(-1,2)
plt.show()

"""
Training KAN
"""
ys = []

# setting bias_trainable=False, sp_trainable=False, sb_trainable=False is important.
# otherwise KAN will have random scaling and shift for samples in previous stages

model = KAN(width=[1,1], grid=200, k=3, noise_scale=0.1, sp_trainable=False, sb_trainable=False, base_fun='zero')

"""for group_id in range(n_peak):
    dataset = {}
    dataset['train_input'] = x_sample[group_id*n_num_per_peak:(group_id+1)*n_num_per_peak][:,None]
    dataset['train_label'] = y_sample[group_id*n_num_per_peak:(group_id+1)*n_num_per_peak][:,None]
    dataset['test_input'] = x_sample[group_id*n_num_per_peak:(group_id+1)*n_num_per_peak][:,None]
    dataset['test_label'] = y_sample[group_id*n_num_per_peak:(group_id+1)*n_num_per_peak][:,None]
    model.fit(dataset, opt = 'LBFGS', steps=100, update_grid=False)
    y_pred = model(x_grid[:,None])
    ys.append(y_pred.detach().numpy()[:,0])"""

dataset = {}
dataset['train_input'] = x_sample[0 * n_num_per_peak:(0 + 1) * n_num_per_peak][:, None]
dataset['train_label'] = y_sample[0 * n_num_per_peak:(0 + 1) * n_num_per_peak][:, None]

mymodel = KAN(width=[1,16,1], grid=200, k=3, noise_scale=0.1, sp_trainable=False, sb_trainable=False, base_fun='zero')
dataset_m = {}
train_inputs = torch.tensor([torch.tensor(float(i)) for i in train_df.index])[:, None]
train_labels = torch.tensor([torch.tensor(float(i)) for i in train_df['LH']])[:, None]
test_inputs = torch.tensor([torch.tensor(float(i)) for i in test_df.index])[:, None]
test_labels = torch.tensor([torch.tensor(float(i)) for i in test_df['LH']])[:, None]
dataset_m['train_input'] = train_inputs
dataset_m['train_label'] = train_labels
dataset_m['test_input'] = test_inputs
dataset_m['test_label'] = test_labels
mymodel.fit(dataset_m, opt ='LBFGS', steps=100, update_grid=False)
my_y_pred = mymodel(train_inputs)
my_ys = my_y_pred.detach().numpy()

plt.plot(train_inputs, train_labels, color='black', alpha=0.1)
plt.plot(train_inputs, my_ys, color='black')
plt.show()


""""# Prediction
plt.subplots(1, 5, figsize=(15, 2))
plt.subplots_adjust(wspace=0, hspace=0)

for i in range(1,6):
    plt.subplot(1,5,i)
    group_id = i - 1
    plt.plot(x_grid.detach().numpy(), y.detach().numpy(), color='black', alpha=0.1)
    plt.plot(x_grid.detach().numpy(), ys[i-1], color='black')
    plt.xlim(-1,1)
    plt.ylim(-1,2)
plt.show()"""