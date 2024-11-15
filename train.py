
# Import libraries important for defining and training the model
import torch
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

# Load weights and biases for logging the training process
import wandb

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


def train_model(experiment_directory, data_source):

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

    # Get the number of epochs
    num_epochs = specs['num_epochs']

    # Define the models
    encoder = Encoder()
    decoder = Decoder()
    time_warper = NeuralODE()

    # Define an optimizer
    optimizer_all = torch.optim.Adam(
          [
              {
                  "params": encoder.parameters(),
                  "lr": lr_schedules[0].get_learning_rate(0),
              },
              {
                  "params": decoder.module.sdf_decoder.parameters(),
                  "lr": lr_schedules[1].get_learning_rate(0),
              },
              {
                  "params": latent_warper.parameters(),
                  "lr": lr_schedules[2].get_learning_rate(0),
              },
          ]
    )

    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_all, gamma=0.99)

    # Start up a weights and biases session
    wandb_dir = os.path.join(experiment_directory, "wandb")
    experiment_name = os.path.basename(experiment_directory)
    if not os.path.isdir(wandb_dir):
        os.makedirs(wandb_dir)
    else:
        raise ValueError("Wandb directory already exists. Remove this directory if one wants to rerun the experiment...")
    wandb.init(project="WassersteinPrior-based-4dImaging", dir=os.path.join(experiment_directory), config=specs,
               notes="The experiment directory is: {}".format(experiment_name))

    # For every epoch, do ...
    for epoch in range(num_epochs):

        # Grab a batch
        for batch in train_dataloader:

            # Encode
            z = ...

            # Latent dynamics
            z_t = ...

            # Decode
            recon = ...

            # Calculate the motion prior loss
            loss_motion_prior = ...

            # Calculate the reconstruction loss
            loss_recon = ...

            # Calculate additional losses
            ...

            # ...


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
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )

    args = arg_parser.parse_args()

    train_model(args.experiment_directory, args.data_source)
