# Import libraries and functions required for calculating the barycenters
import torch
from wasserstein_barycenters import convolutional_barycenter_calculation

# Import plotting libraries
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Import libraries for parsing inputs, for loading things, and for saving things
import os
import argparse
import json

# Import libraries for evaluating the image reconstruction quality
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

# Code related to the datasets that we use
from src.data.datasets import CellData

# Remaining libraries
import numpy as np

def create_time_series_gif(time_series, bary_center_time_series, save_dir, recon_idx):

    # Create a figure
    fig, ax = plt.subplots(1, 2)

    # Calculate the maximum value in the ground truth time series
    max_val = time_series.max()

    ### trajectory animation (gif)
    def animate(i, time_series, bary_center_time_series):
        ax[0].clear()
        ax[1].clear()
        p1 = ax[0].imshow(time_series[i].squeeze(), vmin=0, vmax=max_val)
        p2 = ax[1].imshow(bary_center_time_series[i].squeeze(), vmin=0, vmax=max_val)
        ax[0].set_title("Ground truth at time point {}".format(i))
        ax[1].set_title("Reconstruction at time point {}".format(i))
        return p1, p2

    gif = FuncAnimation(fig, animate, fargs=(time_series, bary_center_time_series),
                        blit=True, repeat=True, frames=time_series.shape[0], interval=1)
    gif.save(os.path.join(save_dir, "BarycenterInterpolatedTimeSeries_{}.gif".format(recon_idx)), dpi=150, writer=PillowWriter(fps=5))
    ax[0].clear()
    ax[1].clear()

    # Close the figures
    plt.close('all')

def evaluate_recon(time_series, time_series_recon, subsampling):

    # We calculate the PSNR, SSIM, and L2 reconstruction metrics along with their variance / standard deviation. We
    # save all the values in a dict of lists.
    metric_list = {}
    metric_list['PSNR'] = []
    metric_list['SSIM'] = []
    metric_list['MSE'] = []

    # For every image in the time series, do ...
    for i in range(time_series.shape[0]):

        # Skip the time series for seen time points
        if i % subsampling == 0:
            continue

        # Else, we calculate the reconstruction metrics
        metric_list['PSNR'].append(peak_signal_noise_ratio(time_series[i].squeeze(), time_series_recon[i].squeeze()))
        metric_list['SSIM'].append(structural_similarity(time_series[i].squeeze(), time_series_recon[i].squeeze(), data_range=1.0))
        metric_list['MSE'].append(mean_squared_error(time_series[i].squeeze(), time_series_recon[i].squeeze()))

    # Return the metric list
    return metric_list

def barycenter_based_interpolation(time_series, subsampling, save_dir, recon_idx, barycenter_type='OT'):

    # First, check the supplied barycenter type
    if not barycenter_type in ['OT', 'l2']:
        raise ValueError("The input 'barycenter_type' is {}, but can only be 'OT' or 'l2'...".format(barycenter_type))

    # Get the subsampled time series
    subsampled_time_series = time_series[::subsampling]

    # Get the time series and get the time points in between
    start_imgs = subsampled_time_series[:-1]
    end_imgs = subsampled_time_series[1:]

    # Get the size of the images
    h, w = start_imgs[0].shape[-2:]

    # Create a grid
    _y, _x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing='ij'
    )

    # Calculate the weights that we want to use
    t_list = list(range(1, subsampling))

    # Make the start images and end images into torch tensors
    start_imgs = torch.tensor(np.array(start_imgs))
    end_imgs = torch.tensor(np.array(end_imgs))

    # Get the batch needed for the barycenter calculation
    batch = torch.stack([start_imgs, end_imgs], dim=0)

    # Copy the original time series
    bary_center_time_series = torch.tensor(np.array(time_series))

    # Normalize the batch
    batch_normalized = batch / torch.sum(batch, dim=(2, 3, 4), keepdim=True)

    # Normalize bary_center_time_series
    bary_center_time_series = bary_center_time_series / torch.sum(bary_center_time_series, dim=(1, 2, 3), keepdim=True)

    # Get the normalize time series as a numpy array
    time_series_normalized = np.array(time_series) / np.sum(np.array(time_series), axis=(1,2,3), keepdims=True)

    # Between every start image and end image, calculate barycenters
    for t in t_list:

        # Calculate the correct type of barycenters
        if barycenter_type == 'OT':

            # Get the correct weight
            weights = torch.tensor([1.0 - t / subsampling, t / subsampling])[:, None, None, None, None]

            # Calculate the OT barycenters with the above weights
            barycenters = convolutional_barycenter_calculation(batch_normalized, weights=weights, stab_thresh=1e-30, scaling=0.99)
        else:
            barycenters = start_imgs * (1.0 - t / subsampling) + end_imgs * (t / subsampling)

        # Add the barycenters at the correct location in the time_series tensor
        bary_center_time_series[t::subsampling] = barycenters

    # Finally, we plot create a gif of the barycenter and save it
    create_time_series_gif(torch.tensor(time_series_normalized), bary_center_time_series, save_dir, recon_idx)

    # Evaluate the reconstruction capability
    metric_list = evaluate_recon(time_series_normalized, bary_center_time_series.numpy(), subsampling)

    # Return the reconstruction values
    return metric_list


def ot_interpolation_time_series(data_source, subsampling, save_dir, barycenter_type):

    # Define the dataset
    dynamic_dataset_test = CellData(data_source, time_step=1, dynamic=True, full_time_series=True)

    # Get the number of time series
    num_time_series = len(dynamic_dataset_test.all_images_per_track)

    # Create a dictionary containing the reconstruction values for each time series
    track_metric_vals = {list(dynamic_dataset_test.all_images_per_track.keys())[i]:
                             barycenter_based_interpolation(dynamic_dataset_test.get_full_track(i),
                                                            subsampling, save_dir,
                                                            list(dynamic_dataset_test.all_images_per_track.keys())[i],
                                                            barycenter_type) for i in range(num_time_series)}

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


if __name__ == "__main__":
    torch.random.manual_seed(31359)
    np.random.seed(31359)

    arg_parser = argparse.ArgumentParser(description="Train")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )

    arg_parser.add_argument(
        "--barycenter_type",
        "-b",
        dest="barycenter_type",
        required=True,
        help="The type of barycenter calculation we use for interpolation. The options are 'OT' and 'l2'. "
    )

    args = arg_parser.parse_args()

    # Get the directory of the current file and via this directory, we go to the main directory
    curr_file_dir = os.path.dirname(__file__)
    main_dir = os.path.join(curr_file_dir, "..", "..")

    specs_filename = os.path.join(main_dir, args.experiment_directory, 'specs.json')

    if not os.path.isfile(specs_filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(args.experiment_directory)
        )

    specs = json.load(open(specs_filename))

    save_dir = os.path.join(main_dir, args.experiment_directory, 'barycentric_interpolations', args.barycenter_type)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    ot_interpolation_time_series(os.path.join(main_dir, specs["DataSource"]), specs["TimeSubsampling"], save_dir, args.barycenter_type)




