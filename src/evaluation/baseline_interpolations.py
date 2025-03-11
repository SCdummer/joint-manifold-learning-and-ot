# Import libraries and functions required for calculating the barycenters
import torch
from .wasserstein_barycenters import convolutional_barycenter_calculation

# Import code used for making visualizations
from src.visualization.create_recon_visualizations import create_time_series_gif

# Import libraries for parsing inputs, for loading things, and for saving things
import os
import argparse
import json

# For calculating the reconstruction metrics
from .utils import create_summary_statistics, evaluate_time_series_recon

# Code related to the datasets that we use
from src.data.datasets import HeLaCellsSuccessive

# Remaining libraries
import numpy as np

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
    create_time_series_gif(torch.tensor(time_series_normalized), bary_center_time_series, save_dir, recon_idx, "BarycenterInterpolatedTimeSeries")

    # Evaluate the reconstruction capability
    metric_list = evaluate_time_series_recon(time_series_normalized, bary_center_time_series.numpy(), subsampling)

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
    create_summary_statistics(track_metric_vals, save_dir)

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




