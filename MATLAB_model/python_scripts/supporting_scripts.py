import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors

def curve_function(x, a, b, c):
    #return a * (c + np.sin(b * x))
    return a * np.sin( (x * (2 * np.pi / (c * 24))) - b )


def fit_sin_curve(train_df, feature, val_df, test_df, original_df):
    x_data = train_df.index.values
    y_data = train_df[feature].values
    x_all = original_df.index.values
    y_all = original_df[feature].values
    popt, pcov = curve_fit(curve_function, x_data, y_data, p0=[1, 1, 25])
    a_opt, b_opt, c_opt = popt
    print(f"Optimal parameters: a={a_opt}, b={b_opt}, c={c_opt}")
    print('a * sin(x * (2*pi/(c*24)) - b)')
    x_fit = np.linspace(min(x_all), max(x_all), 1000)
    y_fit = curve_function(x_fit, *popt)
    plt.plot(train_df.index, train_df[feature], color='black')
    plt.plot(val_df.index, val_df[feature], color='blue')
    plt.plot(test_df.index, test_df[feature], color='red')
    plt.plot(x_fit, y_fit, label='Fitted Curve', color='orange')
    plt.title('Sampled dataframe with raw hours with fitted sin curve')
    plt.ylabel('Time in hours')
    plt.show()


def darken_color(color, factor=0.7):
    # Convert color to RGB if necessary
    if isinstance(color, str):
        color = mcolors.to_rgb(color)  # Convert named color to RGB
    elif isinstance(color, tuple):
        color = mcolors.to_rgb(color)
    darkened_color = tuple(c * factor for c in color)
    return darkened_color


def get_filtered_distances(gt_peaks, pred_peaks, distance_threshold):
    distances = []
    for p in pred_peaks:
        # Find the closest ground truth peak
        closest_gt = np.min(np.abs(gt_peaks - p))
        if closest_gt <= distance_threshold:
            distances.append(closest_gt)
    return np.array(distances)