# Import libraries for defining paths and saving json files
import os
import json

# Remaining libraries
import numpy as np

# Import libraries for evaluating the image reconstruction quality
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

def create_summary_statistics(track_metric_vals, save_dir):
    # Finally, for every metric calculate the average and the standard deviation and save the important values
    metric_statistics = {}
    for metric in ["PSNR", "SSIM", "MSE"]:

        # Initialize the dictionary corresponding to the key 'metric'.
        metric_statistics[metric] = {}

        # Get the values to only the current metric
        metric_vals = [val for track in track_metric_vals.keys() for val in track_metric_vals[track][metric]]

        # Calculate the mean, median, maximum, minimum, and standard deviation
        metric_statistics[metric]['mean'] = np.mean(metric_vals)
        metric_statistics[metric]['median'] = np.median(metric_vals)
        metric_statistics[metric]['max'] = np.max(metric_vals)
        metric_statistics[metric]['min'] = np.min(metric_vals)
        metric_statistics[metric]['std'] = np.std(metric_vals)

    # Save the statistics into a file
    with open(os.path.join(save_dir, "evaluation_metric_statistics.json"), "w") as f:
        json.dump(metric_statistics, f, indent=5)


def evaluate_time_series_recon(time_series, time_series_recon, subsampling=None):

    # We calculate the PSNR, SSIM, and L2 reconstruction metrics along with their variance / standard deviation. We
    # save all the values in a dict of lists.
    metric_list = {}
    metric_list['PSNR'] = []
    metric_list['SSIM'] = []
    metric_list['MSE'] = []

    # For every image in the time series, do ...
    for i in range(time_series.shape[0]):

        # Skip the time series for seen time points
        if subsampling is not None and i % subsampling == 0:
            continue

        # Else, we calculate the reconstruction metrics
        metric_list['PSNR'].append(peak_signal_noise_ratio(time_series[i].squeeze(), time_series_recon[i].squeeze(), data_range=1.0))
        metric_list['SSIM'].append(structural_similarity(time_series[i].squeeze(), time_series_recon[i].squeeze(), data_range=1.0))
        metric_list['MSE'].append(mean_squared_error(time_series[i].squeeze(), time_series_recon[i].squeeze()))

    # Return the metric list
    return metric_list