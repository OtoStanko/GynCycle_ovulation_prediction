import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import tensorflow as tf

import supporting_scripts as sp


class ModelComparator:
    def __init__(self, test_df, input_length, pred_length, features, hormone,
                 step=5, plot=True, peak_comparison_distance=2):
        """
        This class server as a basis for comparison of different models for LH peak prediction.

        :param test_df: dataframe used for testing and subsequently comparing the models
        :param input_length: input length of the data for the models (must be same for all the models)
        :param pred_length: output length for the models (must be same for all the models)
        :param features: list of string, features from the test_df (columns)
        :param hormone: feature from features where peaks should be detected
        :param step: difference between beginning of every consecutive input for comparison. For the best comparison step=1 is advised
        :param plot: if true, plot the individual input-prediction windows. If step is 1 may result in large amount of plots.
        :param peak_comparison_distance: threshold for comparing the peak prediction accuracy. Maximum distance of every predicted peak from the nearest gt peak to be considered truly predicted
        """
        self.test_df = test_df
        self.input_length = input_length
        self.pred_length = pred_length
        self.features = features
        self.num_features = len(features)
        self.hoi_index = features.index(hormone)
        self.hormone = hormone
        self.duration = len(test_df.index)
        self.step = step
        self.plot = plot
        self.peak_comparison_distance = peak_comparison_distance

        self.MIN_PEAK_DISTANCE = 20
        self.MIN_PEAK_HEIGHT = 0.3
        self.results = dict()

        self.peaks_within_threshold = None
        self.peaks_outside_threshold = None
        self.peaks_within_threshold_rev = None
        self.peaks_outside_threshold_rev = None
        self.sum_of_dists_to_nearest_peak = None
        self.num_detected_peaks = None

    def compare_models(self, list_of_models, run_id):
        """
        Compares models based on the data in the ModelComparator.
        ModelComparator can compare models across multiple runs for the final statistics.

        Comparator takes in list of models. For every model computes predictions and identifies peaks in it.
        These peaks are then compared to the ground-truth peaks identified in the test_df.
        ModelComparator tests for every predicted peak if it is within the threshold of the nearest gt peak
        and for every gt peak in the output window if it is with the same threshold of the nearest predicted peak.
        If plotting is enabled, the for every window also plots the models' outputs with detected peaks. Yellow predicted
        peaks are those within the threshold of the nearest gt peak.
        :param list_of_models: list of models to compare
        :param run_id: id of a run
        :return: None
        """
        hormone = self.hormone
        test_df = self.test_df
        input_length = self.input_length
        pred_length = self.pred_length
        # reverse_offset serves as a cutoff of the last records from the testing data. The sliding window will not go
        # over these last days. Peaks from this period are still taken into account for computing the statistics
        # of models. This is just to ensure that there are no outlying predictions that don't have a corresponding
        # ground-truth peak due to the end of the testing data. I advise to use value of last 20 days or so
        reverse_offset = 20

        # Statistics about the model forecast and peaks' predictions
        results = ComparisonResults()

        # Identify peaks in the ground-truth data and plot them
        peaks, _ = scipy.signal.find_peaks(
            self.test_df[hormone], distance=self.MIN_PEAK_DISTANCE / 2, height=self.MIN_PEAK_HEIGHT)
        if self.plot:
            plt.plot(test_df.index, test_df[self.features])
            plt.scatter(test_df.index[peaks], test_df[hormone].iloc[peaks],
                        color='red', zorder=5, label='Highlighted Points')
            plt.xlabel('Time [hours]')
            plt.title('Test {} data'.format(self.features))
            plt.show()
        """
        Move along the testing TS. For every window of input_length + pred_length:
            extract the input data
            make prediction
            extract peaks in the current window (input and output)
            shift peaks by the offset of the current window
        """
        dict_of_model_predictions = {model._name: [] for model in list_of_models}
        batch_size = 32
        i = 0
        limit = self.duration - pred_length - input_length + 1
        while i < limit:
            current_batch_size = min(batch_size, limit - i)
            batch_data = [test_df.iloc[i + j:i + j + self.input_length][self.features].values for j in
                      range(current_batch_size)]
            tensor_batch = tf.convert_to_tensor(batch_data, dtype=tf.float32)
            reshaped_tensor_batch = tf.reshape(tensor_batch, (current_batch_size, self.input_length, self.num_features))
            batch_predictions_dict = {model._name: None for model in list_of_models}
            for model in list_of_models:
                new_tensor = reshaped_tensor_batch[:, :, :model.num_features]
                batch_predictions = model(new_tensor)
                batch_predictions = tf.reshape(batch_predictions, (current_batch_size, self.pred_length, model.num_output_features))
                batch_predictions_dict[model._name] = batch_predictions
                for j in range(current_batch_size):
                    predictions = batch_predictions_dict[model._name][j][:, self.hoi_index]
                    dict_of_model_predictions[model._name].append(predictions)
            i += current_batch_size

        for offset in range(0, self.duration - pred_length - input_length + 1 - reverse_offset, self.step):
            # For every model extract the prediction for this time window
            list_of_model_predictions = []
            for model in list_of_models:
                list_of_model_predictions.append(dict_of_model_predictions[model._name][offset])
            # Ground-truth time in days shifted to start with 0
            gt_time = test_df.index[offset:input_length + pred_length + offset]
            gt_time = gt_time / 24
            first_elem = gt_time[0]
            gt_time = gt_time - first_elem
            # Prediction time in days shifted to start with input_length-th day
            pred_time = test_df.index[input_length + offset:input_length + pred_length + offset]
            pred_time = pred_time / 24
            pred_time = pred_time - first_elem
            # Ground truth values for the whole window
            ground_truth = test_df[hormone][offset:input_length + pred_length + offset]
            # Take only the peaks in the prediction window (input and output window)
            # Shift them so that their time aligns with the offset data
            curr_peaks = np.array([x for x in peaks if offset <= x < input_length + pred_length + offset])
            curr_peaks = curr_peaks - offset
            peaks_for_first_method = np.array([x for x in peaks if offset <= x < input_length + pred_length + offset + reverse_offset])
            peaks_for_first_method = peaks_for_first_method - offset
            gt_peaks_predWindow = np.array([x for x in peaks if offset + input_length <= x < input_length + pred_length + offset])
            gt_peaks_predWindow = gt_peaks_predWindow - offset
            #if len(gt_peaks_predWindow) >= 1:
            #    gt_peaks_predWindow = gt_peaks_predWindow[:1]
            # Try all the peaks, shift them to match the predicted data
            all_peaks_offset = np.array([x for x in peaks]) - offset
            if self.plot:
                plt.plot(gt_time, ground_truth, marker='.', )
            # Plot the tips of the peaks that are in the input-prediction window (input and output window)
            if len(curr_peaks) > 0 and self.plot:
                plt.scatter(gt_time[curr_peaks], ground_truth.iloc[curr_peaks],
                            color='red', zorder=5, label='Test data peaks')
            #methods = ['dense', 'combined', 'raw', 'smooth']
            methods = ['raw' for _ in range(len(list_of_models))]
            for i in range(len(list_of_model_predictions)):
                model = list_of_models[i]
                model_name = model._name
                model_predictions = list_of_model_predictions[i]
                # Detect peaks in the prediction part (forecast) and shift them to start from the right time
                pred_peaks = model.get_peaks(model_predictions, methods[i])
                #pred_peaks, _ = scipy.signal.find_peaks(model_predictions, distance=self.MIN_PEAK_DISTANCE)
                results.num_detected_peaks[model_name] = results.num_detected_peaks.get(model_name, 0) + len(pred_peaks)
                offset_pred_peaks = pred_peaks + input_length
                unfiltered_signed_distances = sp.get_signed_distances(peaks_for_first_method, offset_pred_peaks)
                unfiltered_signed_distances_rev = sp.get_signed_distances(offset_pred_peaks, gt_peaks_predWindow[:1])
                unfiltered_abs_distances = np.array([abs(dist) for dist in unfiltered_signed_distances])
                unfiltered_abs_distances_rev = np.array([abs(dist) for dist in unfiltered_signed_distances_rev])
                # Proceed only if there are any ground-truth peaks in the output part
                if len(curr_peaks) > 0:
                    filtered_distances = np.array(
                        [distance for distance in unfiltered_abs_distances if distance <= self.peak_comparison_distance])
                    results.peaks_within_threshold[model_name] = (
                            results.peaks_within_threshold.get(model_name, 0) + len(filtered_distances))
                    results.peaks_outside_threshold[model_name] = (
                            results.peaks_outside_threshold.get(model_name, 0) + len(pred_peaks) - len(filtered_distances))
                    results.sum_of_dists_to_nearest_peak[model_name] = (
                            results.sum_of_dists_to_nearest_peak.get(model_name, 0) + sum(unfiltered_abs_distances))
                    filtered_distances_rev = np.array(
                        [distance for distance in unfiltered_abs_distances_rev if distance <= self.peak_comparison_distance])
                    results.peaks_within_threshold_rev[model_name] = (
                        results.peaks_within_threshold_rev.get(model_name, 0) + len(filtered_distances_rev))
                    results.peaks_outside_threshold_rev[model_name] = (
                        results.peaks_outside_threshold_rev.get(model_name, 0)
                        + len(gt_peaks_predWindow) - len(filtered_distances_rev))
                    pdd = results.peak_distances_distribution.get(model_name, dict())
                    pddR = results.peak_distances_distribution_rev.get(model_name, dict())
                    for distance in unfiltered_signed_distances:
                        pdd[distance] = pdd.get(distance, 0) + 1
                    for distance in unfiltered_signed_distances_rev:
                        pddR[distance] = pddR.get(distance, 0) + 1
                    results.peak_distances_distribution[model_name] = pdd
                    results.peak_distances_distribution_rev[model_name] = pddR
                if self.plot:
                    line, = plt.plot(pred_time, model_predictions, marker='.', label=model_name)
                    line_color = line.get_color()
                    darker_line_color = sp.darken_color(line_color, 0.5)
                    if len(unfiltered_abs_distances) != 0:
                        for j in range(len(pred_peaks)):
                            if unfiltered_abs_distances[j] <= self.peak_comparison_distance:
                                plt.scatter(pred_time[pred_peaks[j]], model_predictions[pred_peaks[j]],
                                            color='yellow', zorder=5)
                            else:
                                plt.scatter(pred_time[pred_peaks[j]], model_predictions[pred_peaks[j]],
                                            color=darker_line_color, zorder=5)
                    else:
                        if len(pred_peaks) != 0:
                            plt.scatter(pred_time[pred_peaks], model_predictions[pred_peaks],
                                        color=darker_line_color, zorder=5)
            if self.plot:
                plt.axvline(x=input_length, color='r', linestyle='--', )
                plt.legend(loc='upper left')
                plt.title('Prediction on {} days with offset {} days'.format(input_length, offset))
                plt.show()
        self.results[run_id] = results

    def get_run_results(self, run_id):
        """
        Returns ComparisonResults object with results of given run_id. Will throw error if the run_id is invalid.
        :param run_id: id of a run
        :return: ComparisonResults object
        """
        return self.results[run_id]

    def get_run_results_tuple(self, run_id):
        """
        Returns tuple with unwrapped results from ComparisonResults object for the run specified by run_id.
        Will throw error if the run_id is invalid.
        :param run_id: id of a run
        :return: tuple of results in order: peaks_within_threshold, peaks_outside_threshold, peaks_within_threshold_rev,
        peaks_outside_threshold_rev, sum_of_dists_to_nearest_peak, num_detected_peaks, peak_distances_distribution
        """
        results =  self.results[run_id]
        pwt = results.peaks_within_threshold
        pot = results.peaks_outside_threshold
        pwtr = results.peaks_within_threshold_rev
        potr = results.peaks_outside_threshold_rev
        sodtnp = results.sum_of_dists_to_nearest_peak
        ndp = results.num_detected_peaks
        pdd = results.peak_distances_distribution
        return pwt, pot, sodtnp, ndp, pdd, pwtr, potr

    def plot_pred_peak_distribution(self, run_id=None, mode=(True,True)):
        """
        Plots how were predicted peaks distributed around the nearest ground-truth peak. Or (and) how were
        ground-truth peaks distributed around the nearest predicted peak.
        :param run_id: if not specified, all the runs will be plotted.
        :param mode: (plot predicted peaks to gt peaks, plot gt peaks to predicted)
        :return: None
        """
        if len(mode) != 2:
            print("Mode required a tuple of two bools")
            return
        if run_id is None:
            ids_to_plot = [id for id in self.results.keys()]
        else:
            ids_to_plot = [run_id]
        for run_id in ids_to_plot:
            results = self.results.get(run_id, None)
            if results is None:
                print("Wrong id for the results to plot")
                return  # ADD COLOURS BASED ON THE DISTANCE
            peak_distances_distribution = results.peak_distances_distribution
            peak_distances_distribution_rev = results.peak_distances_distribution_rev
            max_val = max(max(max(inner_dict.values()) for inner_dict in peak_distances_distribution.values()),
                          max(max(inner_dict.values()) for inner_dict in peak_distances_distribution_rev.values())) + 1
            for model_name in peak_distances_distribution.keys():
                if mode[0]:
                    pdd = peak_distances_distribution[model_name]
                    keys = list(pdd.keys())
                    values = list(pdd.values())
                    colors = ['yellow' if abs(key) <= self.peak_comparison_distance else '#1f77b4' for key in keys]
                    plt.bar(keys, values, color=colors)
                    plt.xlim(-35, 35)
                    plt.ylim(0, max_val)
                    plt.xlabel('Signed distance of forecasted peaks to the nearest ground truth peak')
                    plt.ylabel('Number of peaks')
                    plt.title('Model name: ' + model_name + " (run ID: {})".format(run_id))
                    plt.show()
                if mode[1]:
                    pddr = peak_distances_distribution_rev[model_name]
                    keys = list(pddr.keys())
                    values = list(pddr.values())
                    colors = ['yellow' if abs(key) <= self.peak_comparison_distance else '#1f77b4' for key in keys]
                    plt.bar(keys, values, color=colors)
                    plt.xlim(-35, 35)
                    plt.ylim(0, max_val)
                    plt.xlabel('Signed distance of ground truth peaks to the nearest forecasted peak')
                    plt.ylabel('Number of peaks')
                    plt.title('Model name: ' + model_name + " (run ID: {})".format(run_id))
                    plt.show()

    def simulation_summary(self):
        """
        This method is called internally by plot_in_out_peaks and print_peak_statistics. The method iterates over the runs
        and extracts info which it concatenates to the summary results. This method basically sets contents of:
        (self.) peaks_within_threshold, peaks_outside_threshold, peaks_within_threshold_rev, peaks_outside_threshold_rev,
        sum_of_dists_to_nearest_peak, num_detected_peaks.
        :return: None
        """
        self.peaks_within_threshold = {}
        self.peaks_outside_threshold = {}
        self.peaks_within_threshold_rev = {}
        self.peaks_outside_threshold_rev = {}
        self.sum_of_dists_to_nearest_peak = {}
        self.num_detected_peaks = {}
        for run_id in self.results.keys():
            within, outside, nearest_dists, num_detected, peak_distances_distribution, within_rev, outside_rev = (
                self.get_run_results_tuple(run_id))

            for model_name, num_peaks_within in within.items():
                self.peaks_within_threshold[model_name] = (
                        self.peaks_within_threshold.get(model_name, list()) + [num_peaks_within])
            for model_name, num_peaks_outside in outside.items():
                self.peaks_outside_threshold[model_name] = (
                        self.peaks_outside_threshold.get(model_name, list()) + [
                    num_peaks_outside])
            for model_name, nearest_peak_dist in nearest_dists.items():
                self.sum_of_dists_to_nearest_peak[model_name] = (
                        self.sum_of_dists_to_nearest_peak.get(model_name, list()) + [
                    nearest_peak_dist])
            for model_name, num_detected_peak in num_detected.items():
                self.num_detected_peaks[model_name] = (
                        self.num_detected_peaks.get(model_name, list()) + [num_detected_peak])
            for model_name, num_peaks_within_rev in within_rev.items():
                self.peaks_within_threshold_rev[model_name] = (
                        self.peaks_within_threshold_rev.get(model_name, list()) + [num_peaks_within_rev])
            for model_name, num_peaks_outside_rev in outside_rev.items():
                self.peaks_outside_threshold_rev[model_name] = (
                        self.peaks_outside_threshold_rev.get(model_name, list()) + [
                    num_peaks_outside_rev])

    def plot_in_out_peaks(self, mode=(True,True)):
        """
        Can plot number of peaks within the threshold vs number of peaks outside the threshold for predicted compared to gt
        and gt compared to predicted. Moreover, percentage of predicted peaks within threshold distance
         of the nearest ground truth peak (PPPWG) vs percentage of ground truth peaks within threshold distance of the nearest
        predicted peak (PGPWP).
        :param mode: tuple of booleans. If mode[0] then plot predicted peaks compared to gt peaks.
        If mode[1] then gt peaks compared to predicted peaks.
        If both, plot also the PPPWG vs PGPWP.
        :return: None
        """
        if len(mode) != 2:
            print("Mode required a tuple of two bools")
            return
        self.simulation_summary()
        if len(self.peaks_within_threshold) == 0:
            print("Comparator data is empty, run compare_models.")
            return
        #colors = mpl.colormaps.get_cmap('tab10')  # Using tab10 colormap with as many colors as there are keys
        colors = mpl.cm.get_cmap('Set3', len(list(self.results[0].peaks_within_threshold.keys())))
        if mode[0] and mode[1]:
            plt.figure(figsize=(8, 6))
            for idx, key in enumerate(self.peaks_within_threshold):
                tp = np.array(self.peaks_within_threshold[key])
                fp = np.array(self.peaks_outside_threshold[key])
                fn = np.array(self.peaks_within_threshold_rev[key])
                tn = np.array(self.peaks_outside_threshold_rev[key])
                print('***************')
                print('Model:', key)
                print('Within:', tp)
                print('Outside:', fp)
                print('Sum:', tp+fp)
                print('WithinRev:', fn)
                print('OutsideRev:', tn)
                print('SumRev:', fn+tn)
                x = tp / (tp + fp)
                y = fn / (fn + tn)
                plt.scatter(x, y, color=colors(idx), label=key)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.plot([0, 1], [1, 0], 'r--',)
            #plt.plot([0, 1], [0.5, 0.5], 'r--',)
            #plt.plot([0.5, 0.5], [0, 1], 'r--',)
            plt.xlabel('How well are the hits')
            plt.ylabel('How well are the peaks hit')
            plt.title('')
            plt.legend(title="Model")
            plt.show()
        if mode[0]:
            plt.figure(figsize=(8, 6))
            for idx, key in enumerate(self.peaks_within_threshold):
                x_values = self.peaks_outside_threshold[key]
                y_values = self.peaks_within_threshold[key]
                # Plot each key's data with a unique color and label it with the key
                plt.scatter(x_values, y_values, color=colors(idx), label=key)

            all_values = itertools.chain(*self.peaks_outside_threshold.values(), *self.peaks_within_threshold.values())
            max_val = max(all_values) + 5
            plt.xlim(0, max_val)
            plt.ylim(0, max_val)
            plt.plot([0, max_val], [0, max_val], 'r--', label='y=x')
            plt.xlabel('# predicted peaks that were further than {} days away from the nearest gt peak'.format(self.peak_comparison_distance))
            plt.ylabel('# predicted peaks that were within {} days of the nearest gt peak'.format(self.peak_comparison_distance))
            plt.title('How well are the prediction peaks placed near the nearest gt peak '
                      '\n(how well placed are the peaks from the prediction)')
            plt.legend(title="Model")
            plt.show()
        if mode[1]:
            plt.figure(figsize=(8, 6))
            for idx, key in enumerate(self.peaks_within_threshold_rev):
                x_values = self.peaks_outside_threshold_rev[key]
                y_values = self.peaks_within_threshold_rev[key]

                # Plot each key's data with a unique color and label it with the key
                plt.scatter(x_values, y_values, color=colors(idx), label=key)

            all_values = itertools.chain(*self.peaks_outside_threshold_rev.values(), *self.peaks_within_threshold_rev.values())
            max_val = max(all_values) + 5
            plt.xlim(0, max_val)
            plt.ylim(0, max_val)
            plt.plot([0, max_val], [0, max_val], 'r--', label='y=x')
            plt.xlabel('# gt peaks that have the nearest predicted peak further than {} days away'.format(self.peak_comparison_distance))
            plt.ylabel('# gt peaks that have the nearest predicted peak within {} days'.format(self.peak_comparison_distance))
            plt.title('How well are the gt peaks predicted by the nearest prediction peak '
                      '\n(how well are the gt peaks identified by the nearest peak from the prediction)')
            plt.legend(title="Model")
            plt.show()

    def print_peak_statistics(self):
        """
        Pretty print for statistics of predicted peaks compared to the nearest ground-truth peak.
        :return: None
        """
        self.simulation_summary()
        if len(self.peaks_within_threshold) == 0:
            print("Comparator data is empty, run compare_models.")
            return
        sp.print_peak_statistics(self.peaks_within_threshold, self.peaks_outside_threshold, self.sum_of_dists_to_nearest_peak,
                                 self.peak_comparison_distance)


class ComparisonResults:
    def __init__(self):
        self.peaks_within_threshold = {}
        self.peaks_outside_threshold = {}
        self.peaks_within_threshold_rev = {}
        self.peaks_outside_threshold_rev = {}
        self.sum_of_dists_to_nearest_peak = {}
        self.num_detected_peaks = {}
        self.peak_distances_distribution = {}
        self.peak_distances_distribution_rev = {}
