# Import libraries and functions required for calculating the barycenters
import torch
from ..training.wasserstein_motion_prior import batched_convol_bary_debiased

# Import plotting libraries
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Import libraries for parsing inputs, for loading things, and for saving things
import os
import argparse
import json

# Code related to the datasets that we use
from src.data.datasets import CellData

# Remaining libraries
import numpy as np

def create_time_series_gif(time_series, save_dir, recon_idx):

    # Create a figure
    fig, ax = plt.subplots(1, 1)

    ### trajectory animation (gif)
    def animate(i, time_series):
        ax.clear()
        p = ax[0].imshow(time_series[i].squeeze(), vmin=0, vmax=1)
        ax.set_title("Reconstruction at time point {}".format(i))
        return p

    gif = FuncAnimation(fig, animate, fargs=time_series,
                        blit=True, repeat=True, frames=time_series.shape[0], interval=1)
    gif.save(os.path.join(save_dir, "BarycenterInterpolatedTimeSeries_{}.gif".format(recon_idx)), dpi=150, writer=PillowWriter(fps=5))
    ax.clear()

    # Close the figures
    plt.close('all')


def barycenter_based_interpolation(time_series, subsampling, save_dir, recon_idx, barycenter_type='OT'):

    # First, check the supplied barycenter type
    if not barycenter_type in ['OT', 'l2']:
        raise ValueError("The input 'barycenter_type' is {}, but can only be 'OT' or 'l2'...".format(barycenter_type))

    # Get the subsampled time series
    idx = [i * subsampling for i in range(time_series.shape[0]) if i * subsampling < time_series.shape[0]]
    subsampled_time_series = time_series[idx]

    # Get the time series and get the time points in between
    start_imgs = subsampled_time_series[:-1]
    end_imgs = subsampled_time_series[1:]

    # Get the size of the images
    h, w = start_imgs.shape[-2:]

    # Create a grid
    _y, _x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing='ij'
    )

    # Calculate the weights that we want to use
    t_list = list(range(1, subsampling))

    # Get the batch needed for the barycenter calculation
    batch = torch.stack([start_imgs, end_imgs], dim=0)

    # Copy the original time series
    bary_center_time_series = time_series.copy()

    # Between every start image and end image, calculate barycenters
    for t in t_list:

        # Calculate the correct type of barycenters
        if barycenter_type == 'OT':

            # Get the correct weight
            weights = torch.tensor([1.0 - t / subsampling, t / subsampling])[:, None].expand(batch.size(1))

            # Calculate the OT barycenters with the above weights
            barycenters = batched_convol_bary_debiased(batch, weights=weights)
        else:
            barycenters = start_imgs * (1.0 - t / subsampling) + end_imgs * (t / subsampling)

        # Add the barycenters at the correct location in the time_series tensor
        bary_center_time_series[t::subsampling] = barycenters

    # Finally, we plot create a gif of the barycenter and save it
    create_time_series_gif(time_series, save_dir, recon_idx)

def ot_interpolation_time_series(data_source, subsampling, save_dir, barycenter_type):

    # Define the dataset
    dynamic_dataset_test = CellData(data_source, time_step=1, dynamic=True, full_time_series=True)

    # Get the time series data
    time_series_dict = dynamic_dataset_test.data_dict

    # For every time series that we have available, do ...
    for track_id, img_list in time_series_dict.items():

        # Get the barycentric interpolation and save it
        barycenter_based_interpolation(img_list, subsampling, save_dir, track_id, barycenter_type)

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

    specs_filename = os.path.join(args.experiment_directory, 'specs.json')

    if not os.path.isfile(specs_filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(args.experiment_directory)
        )

    specs = json.load(open(specs_filename))

    save_dir = os.path.join(args.experiment_directory, 'barycentric_interpolations', args.barycenter_type)

    ot_interpolation_time_series(args.data_source, args.subsampling, save_dir, args.barycenter_type)




