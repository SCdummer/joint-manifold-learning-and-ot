
# Import libraries important for defining and training the model
import torch
import torch.nn as nn
import numpy as np

# Import libraries important for logging, loading, and saving things.
import logging
import argparse
import datetime
import os
import json

# Load the models
from src.models.encoders import Encoder
from src.models.decoders import Decoder
from src.models.latent_warpers import NeuralODE

# Load the wasserstein distance
from src.training.wasserstein_motion_prior import FastConvolutionalW2Cost

# Load the datasets and dataset related functions
from torch.utils.data import DataLoader
from src.data.datasets import CellData

# Load weights and biases for logging the training process
import wandb

# Define some loss functions
recon_loss = nn.MSELoss()
wasserstein_dist = FastConvolutionalW2Cost()

def create_code_snapshot(root, dst_path, extensions=(".py", ".json"), exclude=()):
    """Creates tarball with the source code"""
    import tarfile
    from pathlib import Path

    with tarfile.open(str(dst_path), "w:gz") as tar:
        for path in Path(root).rglob("*"):
            if '.git' in path.parts:
                continue
            exclude_flag = False
            if len(exclude) > 0:
                for k in exclude:
                    if k in path.parts:
                        exclude_flag = True
            if exclude_flag:
                continue
            if path.suffix.lower() in extensions:
                tar.add(path.as_posix(), arcname=path.relative_to(root).as_posix(), recursive=True)


def load_experiment_specifications(experiment_directory):

    filename = os.path.join(experiment_directory, 'specs.json')

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )

    return json.load(open(filename))


def initialize_optimizers_and_schedulers(encoder, decoder, time_warper, init_lr):

    # Define an optimizer
    optimizer_all = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) +
                                 list(time_warper.parameters()), lr=init_lr)

    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_all, gamma=0.99)

    # Return them
    return optimizer_all, scheduler


def get_model_params_dir(experiment_dir, model_name, static_or_dynamic, create_if_nonexistent=False):

    subdir = "static" if static_or_dynamic == 'static' else "dynamic"
    dir = os.path.join(experiment_dir, "ModelParameters", model_name, subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir

def get_optimizer_params_dir(experiment_dir, static_or_dynamic='static', create_if_nonexistent=False):

    subdir = "static" if static_or_dynamic == 'static' else "dynamic"
    dir = os.path.join(experiment_dir, "OptimizerParameters", subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir

def save_model(experiment_directory, filename, model, model_name, static_or_dynamic, epoch):

    model_params_dir = get_model_params_dir(experiment_directory, model_name, static_or_dynamic,True)

    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict()},
        os.path.join(model_params_dir, filename),
    )

def save_optimizer(experiment_directory, filename, optimizer, static_or_dynamic, epoch):

    optimizer_params_dir = get_optimizer_params_dir(experiment_directory, static_or_dynamic,True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )

def save_model_and_optimizer(experiment_directory, static_or_dynamic, epoch, encoder, decoder, time_warper, optimizer, filename='latest.pth'):
    save_model(experiment_directory, filename, decoder, 'decoder', static_or_dynamic, epoch)
    save_model(experiment_directory, filename, encoder, 'encoder', static_or_dynamic, epoch)
    save_model(experiment_directory, filename, time_warper, 'time_warper', static_or_dynamic, epoch)
    save_optimizer(experiment_directory, filename, optimizer, static_or_dynamic, epoch)

def train_model(experiment_directory):

    # Indicate which experiment we are running
    logging.info("Running the experiment specified in the directory: " + experiment_directory)

    # backup the current version of the code
    now = datetime.datetime.now()
    code_bk_path = os.path.join(
        experiment_directory, 'code_bk_%s.tar.gz' % now.strftime('%Y_%m_%d_%H_%M_%S'))
    create_code_snapshot('./', code_bk_path, extensions=('.py', '.json', '.cpp', '.cu', '.h', '.sh'),
                         exclude=('examples', 'third-party', 'bin'))

    # Load the specs.json file
    specs = load_experiment_specifications(experiment_directory)

    # Get the hyperparameters
    num_epochs = specs['num_epochs']
    lambda_dyn_reg = specs['lambda_dyn_reg']
    delta_t = specs['delta_t']
    batch_size = specs['batch_size']
    latent_dim = specs['latent_dim']
    init_lr = specs['init_lr']

    # Get the parameters regarding to saving the models and the optimizers
    save_freq_in_epochs = specs['save_freq']

    # Get the parameters for the encoder, decoder, and latent warper
    encoder_specs = specs['EncoderSpecs']
    decoder_specs = specs['DecoderSpecs']
    time_warper_specs = specs['TimeWarperSpecs']

    # Define the models
    encoder = Encoder(latent_dim, **encoder_specs)
    decoder = Decoder(latent_dim, **decoder_specs)
    time_warper = NeuralODE(latent_dim, **time_warper_specs)

    # Start up a weights and biases session
    wandb_dir = os.path.join(experiment_directory, "wandb")
    experiment_name = os.path.basename(experiment_directory)
    if not os.path.isdir(wandb_dir):
        os.makedirs(wandb_dir)
    else:
        raise ValueError("Wandb directory already exists. Remove this directory if one wants to rerun the experiment...")
    wandb.init(project="WassersteinPrior-based-4dImaging", dir=os.path.join(experiment_directory), config=specs,
               notes="The experiment directory is: {}".format(experiment_name))

    # Create two datasets: a static one and a dynamic one
    static_dataset = CellData(specs["data_source"], time_step=specs["time_step"], dynamic=False)
    dynamic_dataset = CellData(specs["data_source"], time_step=specs["time_step"], dynamic=True)

    # Create the dataloaders
    train_dataloader_static = DataLoader(static_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_dataloader_dynamic = DataLoader(dynamic_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Get the optimizer and the scheduler
    optimizer_all, scheduler = initialize_optimizers_and_schedulers(encoder, decoder, time_warper, init_lr)

    ######################################################################
    ### First, we only train the encoder and decoder on the image data ###
    ######################################################################

    # For every epoch, do ...
    for epoch in range(num_epochs):

        # Grab a batch
        for batch in train_dataloader_static:

            # Encode
            z = encoder(batch)

            # Decode
            recon = decoder(z)

            # Calculate the reconstruction loss
            loss_recon = recon_loss(recon, batch)

            # Backpropagate and update the network parameters
            loss_recon.backward()
            optimizer_all.step()

            # Log the reconstruction loss to weights and biases
            wandb.log({'recon_loss_static': recon_loss.item()})

        # Update the learning rate via the scheduler
        scheduler.step()

        # Save the model
        if epoch % save_freq_in_epochs == 0:
            save_model_and_optimizer(experiment_directory, 'static', epoch, encoder, decoder, time_warper,
                                     optimizer_all, filename='latest.pth')

    ########################################
    ### Now we train everything together ###
    ########################################

    # We reinitialize the optimizers and schedulers
    optimizer_all, scheduler = initialize_optimizers_and_schedulers(encoder, decoder, time_warper, init_lr)

    # For every epoch, do ...
    for epoch in range(num_epochs):

        # Grab a batch
        for (img_1, img_2) in train_dataloader_dynamic:

            # Encode
            z_start = encoder(img_1)
            z_end = encoder(img_2)

            # Latent dynamics
            t = torch.tensor([[0.0, 0.5-delta_t/2, 0.5+delta_t/2, 1.0]]).expand(z_start.size(0), 1)
            z_t = time_warper(z_start, t)

            # Decode everything
            img_1_recon = decoder(z_start)
            img_2_recon_static = decoder(z_end)
            img_2_recon_dynamic = decoder(z_t)

            # Decode
            loss_recon = recon_loss(img_1, img_1_recon) + recon_loss(img_2, img_2_recon_static) + recon_loss(img_2, img_2_recon_dynamic)

            # Normalize the intermediate images for calculating the motion prior loss
            ...

            # Calculate the motion prior loss
            loss_motion_prior = wasserstein_dist(..., ...) / delta_t

            # Calculate the full loss
            loss = loss_recon + lambda_dyn_reg * loss_motion_prior

            # Update the parameters
            loss.backward()
            optimizer_all.step()

            # Log the losses to weights and biases
            wandb.log({'recon_loss_dynamic': recon_loss.item(), 'motion_prior_loss': loss_motion_prior.item()})

        # Update the scheduler
        scheduler.step()

        # Save the model
        if epoch % save_freq_in_epochs == 0:
            save_model_and_optimizer(experiment_directory, 'dynamic', epoch, encoder, decoder, time_warper,
                                     optimizer_all, filename='latest.pth')


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

    args = arg_parser.parse_args()

    train_model(args.experiment_directory)
