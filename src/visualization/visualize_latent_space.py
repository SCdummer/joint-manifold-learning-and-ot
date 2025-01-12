# Load the dimensionality reduction methods for visualizing the latent space
import umap
import phate
from sklearn.decomposition import PCA

# Import things for data loading and encoding
import os
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToTensor

# Import the plotting functionality
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the models
from src.models.encoders import Encoder
# from src.models.decoders import Decoder
from src.models.latent_warpers import NeuralODE

# Get some code that can load the models
from src.utils.loading_and_saving import load_model

# Remaining libraries
import argparse
import json
import random

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42)

def get_latents(dataset_location, encoder, time_warper, nabla_t, num_int_steps):

    # We are going to treat each track separately and save the encodings in a dictionary
    transform = ToTensor()
    encoding_dict = {}
    folders = os.listdir(dataset_location)
    for folder_name in folders:
        track_id = int(folder_name[5:])
        folder = os.path.join(dataset_location, folder_name)
        imgs_t = []
        start_time = 1000000
        end_time = -1000000
        times = []
        for img_filename in sorted(os.listdir(folder)):
            img_path = os.path.join(folder, img_filename)
            imgs_t.append(Image.open(img_path))
            t = int(os.path.splitext(img_filename.split()[0].split("_")[1][1:])[0])
            if t < start_time:
                start_time = t
            elif t > end_time:
                end_time = t
            times.append(t)
        imgs_torch = torch.stack([transform(np.array(img, dtype=np.float32)[..., None]) for img in imgs_t], dim=0)
        imgs_torch = imgs_torch.to(device)
        encoding_dict[track_id] = {}
        if encoder is not None:
            if time_warper is None:
                encoding_dict[track_id]["img_data"] = encoder(imgs_torch)[1].cpu().detach().numpy()
            else:
                z0 = encoder(imgs_torch[0, ...])[1]
                t = torch.linspace(0, (imgs_torch.size(0) - 1) * nabla_t,
                                   (imgs_torch.size(0) - 1) * num_int_steps + 1).to(device)
                encoding_dict[track_id]["img_data"] = time_warper(z0[None, ...], t).cpu().detach().numpy().squeeze()
        else:
            encoding_dict[track_id]["img_data"] = imgs_torch.flatten(start_dim=1).cpu().detach().numpy()
        encoding_dict[track_id]["start_time"] = start_time
        encoding_dict[track_id]["end_time"] = end_time
        encoding_dict[track_id]["times"] = times
    return encoding_dict

def visualize_latent_space(dataset_train_location, dataset_test_location, encoder, time_warper,
                           visualization_type, epoch, num_samples, nabla_t, num_int_steps):

    # Get the latent codes
    encoding_dict_train = get_latents(dataset_train_location, encoder, time_warper, nabla_t, num_int_steps)
    encoding_dict_test = get_latents(dataset_test_location, encoder, time_warper, nabla_t, num_int_steps)

    # Get a random sample of the track_id's
    if num_samples is not None and isinstance(num_samples, int):
        chosen_track_ids = random.sample(list(encoding_dict_train.keys()), num_samples)
        encoding_dict_train_visualize = {track_id: encoding_dict_train[track_id] for track_id in chosen_track_ids}
    else:
        encoding_dict_train_visualize = encoding_dict_train
        chosen_track_ids = list(encoding_dict_train.keys())

    # Get the latent codes and times into array format
    z_t_train = np.concatenate([encoding_dict_train[track_id]["img_data"] for track_id in encoding_dict_train], axis=0)
    z_t_test = np.concatenate([encoding_dict_test[track_id]["img_data"] for track_id in encoding_dict_test], axis=0)
    t_train = np.concatenate([np.array(encoding_dict_train[track_id]["times"]) for track_id in encoding_dict_train], axis=0)
    t_test = np.concatenate([np.array(encoding_dict_test[track_id]["times"]) for track_id in encoding_dict_test], axis=0)
    track_id_train = np.array([track_id for track_id in encoding_dict_train
                               for i in range(encoding_dict_train[track_id]["img_data"].shape[0])], dtype=np.int64)
    track_id_test = np.array([track_id for track_id in encoding_dict_test], dtype=np.int64)
    z_t_train_visualize = np.concatenate([encoding_dict_train_visualize[track_id]["img_data"] for track_id in
                                          encoding_dict_train_visualize], axis=0)
    t_train_visualize = np.concatenate([np.array(encoding_dict_train_visualize[track_id]["times"]) for track_id in
                                        encoding_dict_train_visualize], axis=0)

    # If the latent dimension equals 2, we can just plot the latent dimensions against each other. If not, we have to
    # reduce the dimensionality even more for visualization. This is done via the visualization_type method.
    latent_dim = z_t_train.shape[-1]
    if latent_dim == 2:
        embedding_train = z_t_train
        embedding_train_visualize = z_t_train_visualize
        embedding_test = z_t_test
    else:
        if visualization_type == 'UMAP':
            # TODO: also implement the AlignedUMAP option
            reducer = umap.UMAP(n_neighbors=15, n_components=2)
            reducer.fit(z_t_train)
            embedding_train = reducer.embedding_
            #embedding_train_visualize = reducer.transform(z_t_train_visualize)
            track_in_list = np.in1d(track_id_train, chosen_track_ids)
            embedding_train_visualize = embedding_train[track_in_list]
            t_train_visualize = t_train[track_in_list]
            embedding_test = reducer.transform(z_t_test)
        elif visualization_type == 'PHATE':
            reducer = phate.PHATE(n_components=2, knn=5)
            embedding_train = reducer.fit_transform(z_t_train)
            #embedding_train_visualize = reducer.transform(z_t_train_visualize)
            track_in_list = np.in1d(track_id_train, chosen_track_ids)
            embedding_train_visualize = embedding_train[track_in_list]
            t_train_visualize = t_train[track_in_list]
            embedding_test = reducer.transform(z_t_test)
        elif visualization_type == 'PCA':
            pca = PCA(n_components=2)
            embedding_train = pca.fit_transform(z_t_train)
            #embedding_train_visualize = pca.transform(z_t_train_visualize)
            track_in_list = np.in1d(track_id_train, chosen_track_ids)
            embedding_train_visualize = embedding_train[track_in_list]
            t_train_visualize = t_train[track_in_list]
            embedding_test = pca.transform(z_t_test)
        else:
            raise ValueError("The visualization_type variable can only be 'UMAP', 'PHATE', or 'PCA' and not {}.".format(visualization_type))

    # Plotting the latent space and colored according to time
    fig, ax = plt.subplots()
    t_unique = np.unique(np.concatenate([t_train, t_test], axis=0))
    colors = cm.rainbow(np.linspace(0, 1, t_unique.shape[0]))
    for t, c in zip(t_unique, colors):
        lat_train = embedding_train_visualize[t_train_visualize==t, ...]
        plt.scatter(lat_train[:, 0], lat_train[:, 1], color=c)#, vmin=colors.min(), vmax=colors.max())
    norm = plt.Normalize(np.min(t_unique), np.max(t_unique))
    sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)#plt.cm.get_cmap('rainbow'), norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax)
    plt.show()

    # Save the plot
    save_dir = os.path.join(experiment_directory, "Figures", "Latent plots", str(epoch))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    substring = "dynamic" if time_warper is not None else "static"
    substring = substring if encoder is not None else "image"
    fig.savefig(os.path.join(save_dir, "Latent_space_{}_train_only_{}.png".format(visualization_type, substring)))

    # Close the figures
    plt.close('all')


if __name__ == "__main__":

    # Get the experiment directory
    from pathlib import Path
    torch.random.manual_seed(31359)
    np.random.seed(31359)

    arg_parser = argparse.ArgumentParser(description="Visualize latent space")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        default=Path("./Experiments/Cells"),
        # required=True,
        help="The experiment directory. This directory should include "
             + "experiment specifications in 'specs.json', and logging will be "
             + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--visualization_type",
        "-v",
        dest="visualization_type",
        default="PHATE",
        # required=True,
        help="The type of dimensionality reduction used for visualizing the latent space.",
    )
    arg_parser.add_argument(
        "--num_samples",
        "-n",
        dest="num_samples",
        default=None,
        # required=True,
        help="The number of time series to display in the latent space plot.",
    )
    arg_parser.add_argument('--image_based_plot', dest='image_based_plot',
                            action=argparse.BooleanOptionalAction)
    arg_parser.add_argument('--dynamic', dest='dynamic',
                            action=argparse.BooleanOptionalAction)
    args = arg_parser.parse_args()
    try:
        num_samples = None if args.num_samples is None else int(args.num_samples)
    except:
        raise ValueError(
            "The number of time series displayed option should be None or an integer, not {}".format(args.num_samples))
    experiment_directory = args.experiment_directory

    # Get the directory of the current file and via this directory, we go to the main directory
    curr_file_dir = os.path.dirname(__file__)
    main_dir = os.path.join(curr_file_dir, "..", "..")

    specs_filename = os.path.join(main_dir, experiment_directory, 'specs.json')

    if not os.path.isfile(specs_filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )

    specs = json.load(open(specs_filename))

    save_dir = os.path.join(main_dir, experiment_directory, 'neural_network_reconstructions')

    # Get the latent dimension
    latent_dim = specs["LatentDim"]

    # Get the parameters for the encoder, decoder, and latent warper
    encoder_specs = specs['EncoderSpecs']
    # decoder_specs = specs['DecoderSpecs']
    time_warper_specs = specs['TimeWarperSpecs']

    # Define the models
    if not args.image_based_plot:
        encoder = Encoder(latent_dim, **encoder_specs)
        encoder = encoder.to(device)
        # decoder = Decoder(latent_dim, **decoder_specs, upsample_size=encoder.output_size)
        time_warper = NeuralODE(latent_dim, **time_warper_specs)
        time_warper = time_warper.to(device)

        # Put the models in evaluation mode
        encoder.eval()
        # decoder.eval()
        time_warper.eval()

        # Load the correct parameters
        epoch = load_model(experiment_directory, encoder, 'encoder', 'dynamic', 'latest.pth')
        # load_model(experiment_directory, decoder, 'decoder', 'dynamic', 'latest.pth')
        load_model(experiment_directory, time_warper, 'time_warper', 'dynamic', 'latest.pth')

        if not args.dynamic:
            time_warper = None
    else:
        encoder = None
        time_warper = None
        epoch = 0

    # Get the latent plots
    dataset_train_location = specs["DataSource"]
    dataset_test_location = specs["DataSourceTest"]
    nabla_t = specs["Nabla_t"]
    num_int_steps = specs["NumIntSteps"]
    visualize_latent_space(dataset_train_location, dataset_test_location, encoder, time_warper,
                           args.visualization_type, epoch, num_samples, nabla_t, num_int_steps)
