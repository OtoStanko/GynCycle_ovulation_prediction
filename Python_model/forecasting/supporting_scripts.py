import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def sin_function(x, b, c):
    """
    Sine function in form: 0.05 * sin( (x-b) * (2*pi/(c*24)) ) + 0.05
    :param x: input vector for the sine function
    :param b: shift along the x-axis of the sine function
    :param c: length of the sine function period in days (assumes that the x values are in the hours)
    :return: value of the sine function
    """
    return 0.05 * np.sin( (x-b) * (2 * np.pi / (c * 24)) ) + 0.05


def fit_sin_curve(train_df, feature, val_df, test_df, original_df):
    x_data = train_df.index.values
    y_data = train_df[feature].values
    x_all = original_df.index.values
    y_all = original_df[feature].values
    popt, pcov = curve_fit(sin_function, x_data, y_data, p0=[1, 25])
    b_opt, c_opt = popt
    print(f"Optimal parameters: b={b_opt}, c={c_opt}")
    print('sin(x * (2*pi/(c*24)) - b)')
    x_fit = np.linspace(min(x_all), max(x_all), 1000)
    y_fit = sin_function(x_fit, *popt)
    plt.plot(train_df.index, train_df[feature], color='black')
    plt.plot(val_df.index, val_df[feature], color='blue')
    plt.plot(test_df.index, test_df[feature], color='red')
    plt.plot(x_fit, y_fit, label='Fitted Curve', color='orange')
    plt.title('Sampled dataframe with raw hours with fitted sin curve')
    plt.ylabel('Time in hours')
    plt.show()


def fit_curve(x, y, fun, a=1, b=1, c=1):
    popt, _ = curve_fit(fun, x, y, p0=[a, b, c])
    a_opt, b_opt, c_opt = popt
    return a_opt, b_opt, c_opt


def darken_color(color, factor=0.7):
    """
    Darkens the specified colour by the factor.
    """
    # Convert color to RGB if necessary
    if isinstance(color, str):
        color = mcolors.to_rgb(color)  # Convert named color to RGB
    elif isinstance(color, tuple):
        color = mcolors.to_rgb(color)
    darkened_color = tuple(c * factor for c in color)
    return darkened_color


def get_signed_distances(gt_peaks, pred_peaks):
    """
    For each peak in pred_peaks finds distance to the closest peak in gt_peaks. Positive if the closest gt peak is before
    and negative if the closest gt peak if after the predicted peak.
    :param gt_peaks: list of positions of predicted peaks.
    :param pred_peaks: list of positions of predicted peaks.
    :return: list of signed distances from every predicted peak to the closest gt peak.
    """
    if len(gt_peaks) == 0:
        return np.array([])
    signed_distances = []
    for p in pred_peaks:
        distances = p - gt_peaks
        closest_gt = distances[np.argmin(np.abs(distances))]
        signed_distances.append(closest_gt)
    return np.array(signed_distances)


def print_peak_statistics(peaks_within_threshold, peaks_outside_threshold, sum_of_dists_to_nearest_peak,
                          peak_comparison_distance=2):
    """
    Pretty print for basic statistics of models in peak prediction. Code still works fine, but has mostly been replaced
    by plots and other visualizations.
    :param peaks_within_threshold: dictionary of models' names and lists of numbers of predicted peaks
     within the threshold of the closest gt peak
    :param peaks_outside_threshold: dictionary of model's names and lists of numbers of predicted peaks
     outside the threshold of the closest gt peak
    :param sum_of_dists_to_nearest_peak: dictionary of models' names and lists of sums of distances between
     predicted peaks and the closest gt peak
    :param peak_comparison_distance: threshold for the peak comparison distance
    :return: None
    """
    model_name_length = max([len(key) for key in peaks_within_threshold.keys()] + [10])
    peaks_in_means = dict()
    peaks_in_stds = dict()
    for model_name, peaks_in in peaks_within_threshold.items():
        peaks_in_means[model_name] = round(np.mean(peaks_in), 2)
        peaks_in_stds[model_name] = round(np.std(peaks_in), 2)
    peaks_out_means = dict()
    peaks_out_stds = dict()
    for model_name, peaks_out in peaks_outside_threshold.items():
        peaks_out_means[model_name] = round(np.mean(peaks_out), 2)
        peaks_out_stds[model_name] = round(np.std(peaks_out), 2)
    peaks_in_length = max([len(str(mean))+len(str(std))+4 for mean, std in zip(peaks_in_means.values(), peaks_in_stds.values())] + [16])
    peaks_out_length = max([len(str(mean))+len(str(std))+4 for mean, std in zip(peaks_out_means.values(), peaks_out_stds.values())] + [17])
    thr_width = len(str(peak_comparison_distance))
    distances_means = dict()
    distances_stds = dict()
    for model_name, dists in sum_of_dists_to_nearest_peak.items():
        distances_means[model_name] = round(np.mean(dists), 2)
        distances_stds[model_name] = round(np.std(dists), 2)
    distances_width = max([len(str(mean))+len(str(std))+4 for mean, std in zip(distances_means.values(), distances_stds.values())] + [7])
    print("+-{}-+-{}-+-{}-+-{}-+-{}-+".format(model_name_length * "-", (peaks_in_length + thr_width + 5) * "-",
                                              peaks_out_length * "-", 10 * "-", distances_width * "-"))
    print('| {:{model_width}} | {:{in_width}} (<={:{thr_width}}) | {:{out_width}} | {:{percent_width}} | {:{distances_width}} |'.format(
        f"Model name", f"peaks within thr", peak_comparison_distance, f"peaks outside thr", "Percentage",
            f"SoDttNP",
            model_width=model_name_length, in_width=peaks_in_length, thr_width=thr_width, out_width=peaks_out_length,
            percent_width=10, distances_width=distances_width))
    print("+-{}-+-{}-+-{}-+-{}-+-{}-+".format(model_name_length * "-", (peaks_in_length + thr_width + 5) * "-",
                                         peaks_out_length * "-", 10 * "-", distances_width * "-"))
    for model_name, value in peaks_in_means.items():
        peaks_in_value = str(peaks_in_means[model_name]) + " +- " + str(peaks_in_stds[model_name])
        peaks_out_value = str(peaks_out_means[model_name]) + " +- " + str(peaks_out_stds[model_name])
        distances_value = str(distances_means[model_name]) + " +- " + str(distances_stds[model_name])
        percentage = sum(peaks_within_threshold[model_name]) / (sum(peaks_within_threshold[model_name])+sum(peaks_outside_threshold[model_name]))
        print('| {:{model_width}} | {:{in_width}}    {:{thr_width}}  | {:{out_width}} | {:{percent_width}} | {:{distances_width}} |'.format(
            model_name, peaks_in_value, "", peaks_out_value,
                str(round(percentage, 4)*100)[:5],
                distances_value,
                model_width=model_name_length, in_width=peaks_in_length, thr_width=thr_width,
                out_width=peaks_out_length, percent_width=10, distances_width=distances_width))
    print("+-{}-+-{}-+-{}-+-{}-+-{}-+".format(model_name_length * "-", (peaks_in_length + thr_width + 5) * "-",
                                              peaks_out_length * "-", 10 * "-", distances_width * "-"))
