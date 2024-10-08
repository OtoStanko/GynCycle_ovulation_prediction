import numpy as np
import supporting_scripts as sp
import scipy.signal
import matplotlib.pyplot as plt
import tensorflow as tf

class ModelComparator:
    def __init__(self, test_df, input_length, pred_length, features, hormone,
                 duration=250, step=5, plot=True, peak_comparison_distance=2):
        self.test_df = test_df
        self.input_length = input_length
        self.pred_length = pred_length
        self.features = features
        self.hormone = hormone
        self.duration = duration
        self.step = step
        self.plot = plot
        self.peak_comparison_distance = peak_comparison_distance

        self.MIN_PEAK_DISTANCE = 20
        self.MIN_PEAK_HEIGHT = 0.3
        self.results = dict()

    def compare_models(self, list_of_models, run_id):
        hormone = self.hormone
        test_df = self.test_df
        input_length = self.input_length
        pred_length = self.pred_length

        # Statistics about the model forecast and peaks' predictions
        results = ComparisonResults()

        # Identify peaks in the ground-truth data and plot them
        peaks, _ = scipy.signal.find_peaks(
            self.test_df[hormone], distance=self.MIN_PEAK_DISTANCE / 2, height=self.MIN_PEAK_HEIGHT)
        if self.plot:
            plt.plot(test_df.index, test_df[hormone])
            plt.scatter(test_df.index[peaks], test_df[hormone].iloc[peaks],
                        color='red', zorder=5, label='Highlighted Points')
            plt.xlabel('Time [hours]')
            plt.title('Test {} data'.format(hormone))
            plt.show()
        """
        Move along the testing TS. For every window of input_length + pred_length:
            extract the input data
            make prediction
            extract peaks in the current window (input and output)
            shift peaks by the offset of the current window
        """
        for offset in range(0, self.duration - pred_length - input_length + 1, self.step):
            # Extract input data from the testing df
            inputs = []
            for feature in self.features:
                input = np.array(test_df[feature][offset:input_length + offset], dtype=np.float32)
                tensor = tf.convert_to_tensor(input, dtype=tf.float32)
                inputs.append(tensor)
            tensor_inputs = tf.squeeze(inputs)
            reshaped_tensor = tf.reshape(tensor_inputs, (1, input_length, len(self.features)))
            # For every model make a prediction for this time window
            list_of_model_predictions = []
            for model in list_of_models:
                predictions = model.predict(reshaped_tensor)
                predictions = predictions[0][:, 0]
                list_of_model_predictions.append(predictions)
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
            # Try all the peaks, shift them to match the predicted data
            all_peaks_offset = np.array([x for x in peaks]) - offset
            if self.plot:
                plt.plot(gt_time, ground_truth, marker='.', )
            # Plot the tips of the peaks that are in the input-prediction window (input and output window)
            if len(curr_peaks) > 0 and self.plot:
                plt.scatter(gt_time[curr_peaks], ground_truth.iloc[curr_peaks],
                            color='red', zorder=5, label='Test data peaks')
            for i in range(len(list_of_model_predictions)):
                model = list_of_models[i]
                model_name = model._name
                model_predictions = list_of_model_predictions[i]
                # Detect peaks in the prediction part (forecast) and shift them to start from the right time
                pred_peaks, _ = scipy.signal.find_peaks(model_predictions, distance=self.MIN_PEAK_DISTANCE)
                results.num_detected_peaks[model_name] = results.num_detected_peaks.get(model_name, 0) + len(pred_peaks)
                offset_pred_peaks = pred_peaks + input_length
                unfiltered_signed_distances = sp.get_signed_distances(all_peaks_offset, offset_pred_peaks)
                unfiltered_abs_distances = np.array([abs(dist) for dist in unfiltered_signed_distances])
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
                    pdd = results.peak_distances_distribution.get(model_name, dict())
                    for distance in unfiltered_signed_distances:
                        pdd[distance] = pdd.get(distance, 0) + 1
                    results.peak_distances_distribution[model_name] = pdd
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
                        plt.scatter(pred_time[pred_peaks], model_predictions[pred_peaks],
                                    color=darker_line_color, zorder=5)
            if self.plot:
                plt.axvline(x=input_length, color='r', linestyle='--', )
                plt.legend(loc='upper left')
                plt.title('Prediction on {} days with offset {} days'.format(input_length, offset))
                plt.show()
        self.results[run_id] = results

    def get_run_results(self, run_id):
        return self.results[run_id]

    def get_run_results_tuple(self, run_id):
        results =  self.results[run_id]
        pwt = results.peaks_within_threshold
        pot = results.peaks_outside_threshold
        sodtnp = results.sum_of_dists_to_nearest_peak
        ndp = results.num_detected_peaks
        pdd = results.peak_distances_distribution
        return pwt, pot, sodtnp, ndp, pdd

    def plot_peak_distances(self, run_id=None):
        """
        Plots how were forecasted peaks distributed around the position of ground truth peaks.
        :param run_id: if not specified, all the runs will be plotted.
        :return: None
        """
        if run_id is None:
            ids_to_plot = [id for id in self.results.keys()]
        else:
            ids_to_plot = [run_id]
        for run_id in ids_to_plot:
            results = self.results.get(run_id, None)
            if results is None:
                print("Wrong id for the results to plot")
                return
            peak_distances_distribution = results.peak_distances_distribution
            for model_name, pdd in peak_distances_distribution.items():
                keys = list(pdd.keys())
                values = list(pdd.values())
                plt.bar(keys, values)
                plt.xlim(-35, 35)
                plt.xlabel('Signed distance of forecasted peaks to the nearest ground truth peak')
                plt.ylabel('Number of peaks')
                plt.title('Model name: ' + model_name + " (run ID: {})".format(run_id))
                plt.show()


class ComparisonResults:
    def __init__(self):
        self.peaks_within_threshold = {}
        self.peaks_outside_threshold = {}
        self.sum_of_dists_to_nearest_peak = {}
        self.num_detected_peaks = {}
        self.peak_distances_distribution = {}
