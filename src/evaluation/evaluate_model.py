# Import libraries and functions required for calculating the barycenters
import torch

from sklearn.decomposition import PCA

# Import code for creating visualization
from src.visualization.create_recon_visualizations import create_time_series_gif

# Import libraries for parsing inputs, for loading things, and for saving things
import os
import argparse
import json

# Code related to the datasets that we use
from src.data.datasets import HeLaCellsSuccessive

# Remaining libraries
import numpy as np

# Load the models
from src.models.encoders import Encoder
from src.models.decoders import Decoder
from src.models.latent_warpers import NeuralODE

# Get some code that can load the models
from src.utils.loading_and_saving import load_model

# Import code for calculating (and saving) reconstruction metrics
from .utils import create_summary_statistics, evaluate_time_series_recon

# Import matplotlib for saving things
import matplotlib.pyplot as plt


def save_individual_images(time_series_recon, save_dir, max_val):
    if max_val == "time_series":
        max_val = time_series_recon.max()
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for i in range(time_series_recon.shape[0]):
        fig, ax = plt.subplots()
        plt.imshow(time_series_recon[i, ...].squeeze().astype(np.float32) / max_val, vmin=0, vmax=1.0)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(save_dir, 'recon_' + str(i) + '.png'), bbox_inches='tight',
                    pad_inches=0)
        plt.close('all')


def evaluate_model_on_time_series(encoder, decoder, time_warper, time_series_input, time_series_gt, save_dir, recon_idx,
                                  nabla_t,
                                  num_int_steps, max_val="time_series", recon_type='static', time_subsampling=None, n_successive=None):
    
    # Make sure the max_val input is either a fixed float or equal to "time_series"
    if not (max_val == "time_series"):
        try: 
            max_val = float(max_val)
        except ValueError:
            print("max_val can be a float or is equal to 'time_series'...")

    # Only do encoding-decoding when you want the static reconstructions. Else use time_warper for getting the latents
    if recon_type == 'static':

        # Get the latents
        _, z, _ = encoder(time_series_input)

        # Get the reconstructions
        time_series_recon = decoder(z)

        # Get the evaluation metrics
        metric_list = evaluate_time_series_recon(time_series_gt.cpu().detach().numpy(),
                                                 time_series_recon.cpu().detach().numpy())

        # Save the created time series
        create_time_series_gif(time_series_gt.cpu().detach().numpy(), time_series_recon.cpu().detach().numpy(),
                               save_dir, recon_idx, "StaticReconstructions")

        # Save the individual images
        save_individual_images(time_series_recon.cpu().detach().numpy(),
                               os.path.join(save_dir, "Track_{}".format(recon_idx)), max_val)

    elif recon_type == 'dynamic':

        # Get the latents of the initial image
        _, z0, _ = encoder(time_series_input[0][None, ...])

        # Get a latent time series
        t = torch.linspace(0, (time_series_input.size(0) - 1) * nabla_t,
                           (time_series_input.size(0) - 1) * num_int_steps + 1)
        z_t = time_warper(z0.reshape(1, -1),
                          t)  # .reshape((time_series.size(0) - 1) * num_int_steps + 1, -1, z0.size(1), encoder.output_size[0], encoder.output_size[1])

        # Get the points of reconstruction
        z_t = z_t[::num_int_steps].squeeze()

        # Apply the decoder to get a reconstructed time series
        time_series_recon = decoder(z_t)

        # Get the evaluation metrics
        metric_list = evaluate_time_series_recon(time_series_gt.cpu().detach().numpy(),
                                                 time_series_recon.cpu().detach().numpy())

        # Save the created time series
        create_time_series_gif(time_series_gt.cpu().detach().numpy(), time_series_recon.cpu().detach().numpy(),
                               save_dir, recon_idx, "DynamicReconstructions")

        # Save the individual images
        save_individual_images(time_series_recon.cpu().detach().numpy(),
                               os.path.join(save_dir, "Track_{}".format(recon_idx)), max_val)

        # if encoder latent dimension is 2, we can plot the latent space learnt by the time warper
        latents = z_t.cpu().detach().numpy()
        if not encoder.latent_dim == 2:
            pca = PCA(n_components=2)
            pca.fit(latents)
            latents = pca.transform(latents)

        fig, ax = plt.subplots(figsize=(10, 10))
        
        # plot the points, and mark with a dot each time point with a small circle and frame number
        for i, (x, y) in enumerate(latents):
            ax.plot(x, y, 'o', markersize=3)
            ax.text(x, y, str(i), fontsize=8)
        
        # connect the dots
        ax.plot(latents[:, 0], latents[:, 1], 'k-')
        
        # save the plot with as little white space as possible
        plt.tight_layout()
        _latent_save_dir = os.path.join(save_dir, "Latent_space")
        if not os.path.isdir(_latent_save_dir):
            os.makedirs(_latent_save_dir)
        plt.savefig(os.path.join(_latent_save_dir, f"Latent_space_{recon_idx}.png"))
        plt.close('all')

    return metric_list, latents if recon_type == 'dynamic' else None


def evaluate_model_on_full_dataset(experiment_directory, specs, save_dir, split='train', max_val="time_series"):
    # Get the latent dimension
    latent_dim = specs["LatentDim"]

    # Get the parameters for the encoder, decoder, and latent warper
    encoder_specs = specs['EncoderSpecs']
    decoder_specs = specs['DecoderSpecs']
    time_warper_specs = specs['TimeWarperSpecs']

    # Define the models
    encoder = Encoder(latent_dim, **encoder_specs)
    decoder = Decoder(latent_dim, **decoder_specs, upsample_size=encoder.output_size)
    time_warper = NeuralODE(latent_dim, **time_warper_specs)

    # Put the models in evaluation mode
    encoder.eval()
    decoder.eval()
    time_warper.eval()

    # Load the correct parameters 
    load_model(experiment_directory, encoder, 'encoder', 'dynamic', 'latest.pth')
    load_model(experiment_directory, decoder, 'decoder', 'dynamic', 'latest.pth')
    load_model(experiment_directory, time_warper, 'time_warper', 'dynamic', 'latest.pth')

    # Define the dataset
    data_source = os.path.join(os.path.dirname(__file__), "..", "..", specs["DataSource"])
    test_data_source = os.path.join(os.path.dirname(__file__), "..", "..",
                                    specs.get("TestDataSource", specs["DataSource"]))
    dynamic_dataset_input = HeLaCellsSuccessive(data_source, seed=42, test_size=0.2, split=split, full_time_series=True)
    dynamic_dataset_gt = HeLaCellsSuccessive(test_data_source, seed=42, test_size=0.2, split=split, full_time_series=True)

    # Get the number of time series
    num_time_series = len(dynamic_dataset_input.all_images_per_track)

    # Create a dictionary containing the reconstruction values for each time series
    if not os.path.isdir(os.path.join(save_dir, 'Static reconstructions')):
        os.mkdir(os.path.join(save_dir, 'Static reconstructions'))
    track_metric_vals_static = {list(dynamic_dataset_input.all_images_per_track.keys())[i]:
                                      evaluate_model_on_time_series(encoder, decoder, time_warper,
                                                                    torch.tensor(np.array(dynamic_dataset_input.get_full_track(i))),
                                                                    torch.tensor(np.array(dynamic_dataset_gt.get_full_track(i))),
                                                                    os.path.join(save_dir, 'Static reconstructions'),
                                                                    i, specs["Nabla_t"], specs["NumIntSteps"],
                                                                    max_val,
                                                                    'static') for i in range(num_time_series)}

    if not os.path.isdir(os.path.join(save_dir, 'Dynamic reconstructions')):
        os.mkdir(os.path.join(save_dir, 'Dynamic reconstructions'))
    track_metric_vals_dynamic = {
        list(dynamic_dataset_input.all_images_per_track.keys())[i]:
            evaluate_model_on_time_series(
                encoder, decoder, time_warper,
                torch.tensor(np.array(dynamic_dataset_input.get_full_track(i))),
                torch.tensor(np.array(dynamic_dataset_gt.get_full_track(i))),
                os.path.join(save_dir, 'Dynamic reconstructions'),
                i, specs["Nabla_t"], specs["NumIntSteps"],
                max_val,
                'dynamic'
            ) for i in range(num_time_series)
    }

    metrics = {}
    all_latents = {}
    for key, (metric_list, latents) in track_metric_vals_dynamic.items():
        metrics[key] = metric_list
        if latents is not None:
            all_latents[key] = latents
            
    metrics_static = {}
    for key, (metric_list, _) in track_metric_vals_static.items():
        metrics_static[key] = metric_list

    all_latents_np = np.concatenate([latents for latents in all_latents.values()], axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_latents_np)
    all_latents = {key: pca.transform(latents) for key, latents in all_latents.items()}
    # plot the latents and save them
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(num_time_series):
        track_key = list(dynamic_dataset_input.all_images_per_track.keys())[i]
        latents = all_latents[track_key]
        ax.plot(latents[:, 0], latents[:, 1], label=i)
    ax.legend()
    ax.set_title('Latent space learnt by the time warper')
    _save_dir_for_latent = os.path.join(save_dir, "Dynamic reconstructions", "Latent_space")
    if not os.path.isdir(_save_dir_for_latent):
        os.makedirs(_save_dir_for_latent)
    plt.savefig(os.path.join(_save_dir_for_latent, "all_latent_space.png"))
    plt.close('all')

    # Finally, for every metric calculate the average and the standard deviation and save the important values
    create_summary_statistics(metrics_static, os.path.join(save_dir, 'Static reconstructions'))
    create_summary_statistics(metrics, os.path.join(save_dir, 'Dynamic reconstructions'))
    

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
        "--max_val",
        "-m",
        dest="max_val",
        default="time_series",
        help="The maximum value that we use for plotting the images (not used in the gifs)",
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

    save_dir = os.path.join(main_dir, args.experiment_directory, 'neural_network_reconstructions')

    # Evaluation on the training set
    if not os.path.isdir(os.path.join(save_dir, 'train')):
        os.makedirs(os.path.join(save_dir, 'train'))
    evaluate_model_on_full_dataset(args.experiment_directory, specs, os.path.join(save_dir, 'train'), split='train', max_val=args.max_val)

    # Evaluation on the test set
    if not os.path.isdir(os.path.join(save_dir, 'test')):
        os.makedirs(os.path.join(save_dir, 'test'))
    evaluate_model_on_full_dataset(args.experiment_directory, specs, os.path.join(save_dir, 'test'), split='test', max_val=args.max_val)