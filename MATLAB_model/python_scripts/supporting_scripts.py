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


def get_distances(gt_peaks, pred_peaks):
    if len(gt_peaks) == 0:
        return np.array([])
    unfiltered_distances = []
    for p in pred_peaks:
        # Find the closest ground truth peak
        closest_gt = np.min(np.abs(gt_peaks - p))
        unfiltered_distances.append(closest_gt)
    return np.array(unfiltered_distances)


def print_peak_statistics(peaks_within_threshold, peaks_outside_threshold, sum_of_dists_to_nearest_peak,
                          peak_comparison_distance=2):
    model_name_length = max([len(key) for key in peaks_within_threshold.keys()] + [10])
    #model_name_length = max(model_name_length, 10)
    peaks_in_length = max([len(str(value)) for value in peaks_within_threshold.values()] + [16])
    peaks_out_length = max([len(str(value)) for value in peaks_outside_threshold.values()] + [17])
    thr_width = len(str(peak_comparison_distance))
    distances_width = max([len(str(value)) for value in sum_of_dists_to_nearest_peak.values()] + [7])
    print("+-{}-+-{}-+-{}-+-{}-+-{}-+".format(model_name_length * "-", (peaks_in_length + thr_width + 5) * "-",
                                              peaks_out_length * "-", 10 * "-", distances_width * "-"))
    print('| {:{model_width}} | {:{in_width}} (<={:{thr_width}}) | {:{out_width}} | {:{percent_width}} | {:{distances_width}} |'.format(
        f"Model name", f"peaks within thr", peak_comparison_distance, f"peaks outside thr", "Percentage",
            f"SoDttNP",
            model_width=model_name_length, in_width=peaks_in_length, thr_width=thr_width, out_width=peaks_out_length,
            percent_width=10, distances_width=distances_width))
    print("+-{}-+-{}-+-{}-+-{}-+-{}-+".format(model_name_length * "-", (peaks_in_length + thr_width + 5) * "-",
                                         peaks_out_length * "-", 10 * "-", distances_width * "-"))
    for model_name, value in peaks_within_threshold.items():
        print('| {:{model_width}} | {:{in_width}}    {:{thr_width}}  | {:{out_width}} | {:{percent_width}} | {:{distances_width}} |'.format(
            model_name, value, "", peaks_outside_threshold[model_name],
                str(round(value/(value+peaks_outside_threshold[model_name]), 4)*100)[:5],
                str(sum_of_dists_to_nearest_peak[model_name]),
                model_width=model_name_length, in_width=peaks_in_length, thr_width=thr_width,
                out_width=peaks_out_length, percent_width=10, distances_width=distances_width))
    print("+-{}-+-{}-+-{}-+-{}-+-{}-+".format(model_name_length * "-", (peaks_in_length + thr_width + 5) * "-",
                                              peaks_out_length * "-", 10 * "-", distances_width * "-"))
#print_peak_statistics({"key 1": 5, "a very long key": 4}, {"key 1": 2, "a very long key": 4}, 10)