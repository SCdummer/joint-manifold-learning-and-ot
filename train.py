# Import libraries important for defining and training the model
import torch
import torch.nn as nn
import numpy as np

# Import libraries important for logging, loading, and saving things.
import logging
import argparse
import datetime
import os
from tqdm.auto import tqdm

# Import own created saving and loading functions
from src.utils.loading_and_saving import (
    create_code_snapshot, load_experiment_specifications, save_model_and_optimizer, load_model, load_optimizer
)

# Load the models
from src.models.encoders import Encoder
from src.models.decoders import Decoder
from src.models.latent_warpers import NeuralODE

# Load the wasserstein distance
from src.training.wasserstein_motion_prior import FastConvolutionalW2Cost

# Load the datasets and dataset related functions
from torch.utils.data import DataLoader
from src.data.datasets import HeLaCellsSuccessive

# Load weights and biases for logging the training process
import wandb

# Import code for evaluating the model
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# For rearranging tensors
from einops import rearrange

# Define some loss functions
dataset_class = HeLaCellsSuccessive
recon_loss = nn.MSELoss()  # nn.MSELoss() #nn.L1Loss() # nn.MSELoss() # nn.BCELoss()
wasserstein_dist = FastConvolutionalW2Cost()


def initialize_optimizers_and_schedulers(encoder, decoder, time_warper, init_lr, joint_learning=True):
    # Define an optimizer
    if joint_learning:
        optimizer_all = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) +
                                         list(time_warper.parameters()), lr=init_lr)
    else:
        optimizer_all = torch.optim.Adam(time_warper.parameters(), lr=init_lr)

    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_all, gamma=0.9955)

    # Return them
    return optimizer_all, scheduler


def create_time_series_gif(recon_tensor, gt_tensor, save_dir, recon_idx):
    # Create a figure
    fig, ax = plt.subplots(1, 2)

    ### trajectory animation (gif)
    def animate(i, recon_tensor, gt_tensor):
        ax[0].clear()
        ax[1].clear()
        p1 = ax[0].imshow(recon_tensor[i].squeeze(), vmin=0, vmax=1)
        p2 = ax[1].imshow(gt_tensor[i].squeeze(), vmin=0, vmax=1)
        ax[0].set_title("Reconstruction at time point {}".format(i))
        ax[1].set_title("Ground Truth at time point {}".format(i))
        return p1, p2

    gif = FuncAnimation(fig, animate, fargs=(recon_tensor, gt_tensor),
                        blit=True, repeat=True, frames=gt_tensor.shape[0], interval=1)
    gif.save(os.path.join(save_dir, "ReconstructionTimeSeries_{}.gif".format(recon_idx)), dpi=150,
             writer=PillowWriter(fps=5))
    ax[0].clear()
    ax[1].clear()

    # Close the figures
    plt.close('all')


def change_model_mode(encoder, decoder, time_warper, new_mode):
    """"
    This code puts the models in training or evaluation mode
    """
    if new_mode == 'train':
        encoder.train()
        decoder.train()
        time_warper.train()
    elif new_mode == 'eval':
        encoder.eval()
        decoder.eval()
        time_warper.eval()
    else:
        raise ValueError("Input 'new_mode' can only be 'train' or 'eval' and not {}...".format(new_mode))


def evaluate_random_dynamic_train_reconstructions(
        experiment_directory, encoder, decoder, time_warper, dataloader,
        nabla_t, num_int_steps, time_subsampling, epoch, device, p_bar, metrics_dict
):
    # Change the mode of the models to evaluation
    change_model_mode(encoder, decoder, time_warper, 'eval')

    # Create a figure for each of the inputs and save it to the correct directory
    for i, (time_series, _) in enumerate(dataloader):

        # Put the time series on cuda
        time_series = time_series.to(device)

        # Get the initial image
        image0 = time_series[0, ...][None, ...]

        # Encode the initial image
        _, z0, _ = encoder(image0)

        # Get a latent time series
        t = torch.linspace(0, (time_series.size(0) - 1) * nabla_t, (time_series.size(0) - 1) * num_int_steps + 1).to(device)
        z_t = time_warper(z0.reshape(1, -1), t)
        # .reshape((time_series.size(0) - 1) * num_int_steps + 1, -1, z0.size(1), encoder.output_size[0], encoder.output_size[1])

        # Get the points of reconstruction
        z_t = z_t[::num_int_steps].squeeze()

        # Get reconstructions of the time series
        recon = decoder(z_t)

        # Get the substring for saving the figure
        substring = "dynamic"

        # Create the save directory if it does not exist yet and save a gif for the time series reconstruction
        save_dir = os.path.join(experiment_directory, "Figures", substring, str(epoch))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        create_time_series_gif(recon.detach().cpu().numpy(), time_series.detach().cpu().numpy(), save_dir, i)

        # Also check the one step predictions

        # Get the reconstructions as done via the training procedure. Note: here we need to use the subsampling!
        images_start = time_series[:-time_subsampling:time_subsampling]
        _, z_start, _ = encoder(images_start)
        end_time = nabla_t * time_subsampling
        t = torch.linspace(0.0, end_time, num_int_steps * time_subsampling + 1).to(device)
        z_end = time_warper(z_start.reshape(images_start.size(0), -1), t)[-1]
        # .reshape(time_subsampling * num_int_steps + 1, -1,
        # z_start.size(1), encoder.output_size[0],
        # encoder.output_size[1])[-1]
        images_end = time_series[time_subsampling::time_subsampling, ...]
        images_end_recon = decoder(z_end)
        images_start_recon = decoder(z_start)

        # Get the loss value
        losses_start = (torch.sum((images_start - images_start_recon) ** 2, dim=(1, 2, 3)) / (
                images_start_recon.shape[1] * images_start_recon.shape[2] * images_start_recon.shape[3]))
        losses_end = (torch.sum((images_end - images_end_recon) ** 2, dim=(1, 2, 3)) / (
                images_end_recon.shape[1] * images_end_recon.shape[2] * images_end_recon.shape[3]))

        # Create a figure for each of the inputs and save it to the correct directory
        for j in range(images_end.size(0) - 1):

            # Generate a figure
            fig, ax = plt.subplots(2, 2)
            ax[0, 0].imshow(images_start[j, ...].detach().cpu().numpy().squeeze(), vmin=0, vmax=1)
            ax[0, 1].imshow(images_end[j, ...].detach().cpu().numpy().squeeze(), vmin=0, vmax=1)
            ax[1, 0].imshow(images_start_recon[j, ...].detach().cpu().numpy().squeeze(), vmin=0, vmax=1)
            ax[1, 1].imshow(images_end_recon[j, ...].detach().cpu().numpy().squeeze(), vmin=0, vmax=1)
            ax[0, 0].set_title("Ground truth (start)")
            ax[0, 1].set_title("Ground truth (end)")
            ax[1, 0].set_title("Reconstruction (start) (loss: {:.5f})".format(losses_start[j].item()))
            ax[1, 1].set_title("Reconstruction (end) (loss: {:.5f})".format(losses_end[j].item()))

            # Change the spacing in the subplots
            fig.subplots_adjust(wspace=.75)

            # Get the substring for saving the figure
            substring = "dynamic"

            # Create the save directory if it does not exist yet
            save_dir = os.path.join(
                experiment_directory, "Figures", substring, str(epoch), 'single_time_step_recon', 'time_series_{}'.format(i)
            )
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            fig.savefig(os.path.join(save_dir, "Reconstruction_t{}.png".format(j)))

            # Close the figures
            plt.close('all')

        # Also save the static reconstructions
        _, z, _ = encoder(time_series)
        recon_static = decoder(z)

        # Get the loss value
        losses = (
                torch.sum((recon_static - time_series) ** 2, dim=(1, 2, 3)) / (
                recon_static.shape[1] * recon_static.shape[2] * recon_static.shape[3])
        )

        # Create a figure for each of the inputs and save it to the correct directory
        for k in range(recon_static.size(0)):

            # Generate a figure
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(time_series[k, ...].detach().cpu().numpy().squeeze(), vmin=0, vmax=1)
            ax[1].imshow(recon_static[k, ...].detach().cpu().numpy().squeeze(), vmin=0, vmax=1)
            ax[0].set_title("Ground truth")
            ax[1].set_title("Reconstruction (loss: {:.5f})".format(losses[k].item()))

            # Create the save directory if it does not exist yet
            save_dir = os.path.join(experiment_directory, "Figures", 'dynamic', str(epoch), 'static recons',
                                    'time_series_{}'.format(i))
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            fig.savefig(os.path.join(save_dir, "Reconstruction_time_series_t{}.png".format(k)))

            # Close the figures
            plt.close('all')

        # Log the losses to weights and biases
        metrics_dict.update({
            'test/losses': losses.mean().item()
        })
        p_bar.set_postfix(metrics_dict)

        if i == 4:
            break

    # Change the mode of the models back to train
    change_model_mode(encoder, decoder, time_warper, 'train')


def evaluate_random_static_train_reconstructions(
        experiment_directory, encoder, decoder, dataloader, epoch, device, num_samples=5
):
    # Get a batch of the training data
    inputs, _ = next(iter(dataloader))
    inputs = inputs.to(device)

    # Get some random number of samples
    if inputs.size(0) > num_samples:
        perm = torch.randperm(inputs.size(0))
        idx = perm[:num_samples]
        inputs = inputs[idx]

    # Encode and decode them
    z, _, _ = encoder(inputs)
    recon = decoder(z)

    # Get the loss value
    losses = (torch.sum((recon - inputs) ** 2, dim=(1, 2, 3)) / (recon.shape[1] * recon.shape[2] * recon.shape[3]))

    # Create a figure for each of the inputs and save it to the correct directory
    for i in range(recon.size(0)):

        # Generate a figure
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(inputs[i, ...].detach().cpu().numpy().squeeze(), vmin=0, vmax=1)
        ax[1].imshow(recon[i, ...].detach().cpu().numpy().squeeze(), vmin=0, vmax=1)
        ax[0].set_title("Ground truth")
        ax[1].set_title("Reconstruction (loss: {:.5f})".format(losses[i].item()))

        # Get the substring for saving the figure
        substring = "static"

        # Create the save directory if it does not exist yet
        save_dir = os.path.join(experiment_directory, "Figures", substring, str(epoch))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fig.savefig(os.path.join(save_dir, "Reconstruction_{}.png".format(i)))

        # Close the figures
        plt.close('all')


def full_time_series_collate(batch):
    return [torch.from_numpy(time_series) for time_series in batch]


def plot_time_series(track_id, dataset):
    # Get the time series
    time_series = dataset.data_dict[track_id]

    # For every time step in the time series, do ...
    for i in range(len(time_series)):
        plt.imshow(time_series[i].squeeze(), vmin=0, vmax=1)
        plt.title("(max, min) = ({}, {})".format(time_series[i].max(), time_series[i].min()))
        plt.show()


recon_loss_no_reduc = torch.nn.MSELoss(reduction='none')


def vae_recon_loss(img, img_recon, mu_z, log_var_z):
    recon_loss_val = 0.5 * recon_loss_no_reduc(img.reshape(img.shape[0], -1),
                                               img_recon.reshape(img_recon.shape[0], -1)).sum(dim=-1)
    KLD = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp(), dim=-1)
    weighting_factor = 1.0 / (img.reshape(img.shape[0], -1).shape[-1])
    return (recon_loss_val + 0.005 * KLD).mean(dim=0) * weighting_factor, recon_loss_val.mean(
        dim=0) * weighting_factor, KLD.mean(dim=0) * weighting_factor


def grad_encoder_loss(img, z):
    grad = torch.autograd.grad(outputs=z,
                               inputs=img,
                               grad_outputs=torch.ones_like(z).to(z.device),
                               create_graph=True,
                               retain_graph=True)[0].reshape(z.shape[0], -1)
    return (torch.norm(grad, dim=-1) ** 2).mean()


def train_model(experiment_directory):
    # First, check if the experiment directory exists. If not, indicate this to the user
    if not os.path.isdir(experiment_directory):
        raise ValueError("The experiment directory {} does not exist! "
                         "Please make sure you are pointing to a valid directory...".format(experiment_directory))

    # Use cuda if cuda is available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Indicate which experiment we are running
    logging.info("Running the experiment specified in the directory: " + str(experiment_directory))

    # backup the current version of the code if there is not backup
    backup_code = True
    for object_in_folder in os.listdir(experiment_directory):
        if object_in_folder.startswith("code_bk_"):
            backup_code = False
            break
    if backup_code:
        now = datetime.datetime.now()
        code_bk_path = os.path.join(
            experiment_directory, 'code_bk_%s.tar.gz' % now.strftime('%Y_%m_%d_%H_%M_%S'))
        create_code_snapshot('./', code_bk_path, extensions=('.py', '.json', '.cpp', '.cu', '.h', '.sh'),
                             exclude=('examples', 'third-party', 'bin'))

    # Load the specs.json file
    specs = load_experiment_specifications(experiment_directory)

    # Get the hyperparameters
    num_epochs_static = specs['NumEpochsStatic']
    num_epochs_dynamic = specs['NumEpochsDynamic']
    h = specs['h']
    nabla_t = specs["Nabla_t"]
    batch_size_static = specs['BatchSizeStatic']
    batch_size_dynamic = specs["BatchSizeDynamic"]
    latent_dim = specs['LatentDim']
    init_lr = specs['InitLR']
    num_int_steps = specs['NumIntSteps']
    time_subsampling = specs["TimeSubsampling"]
    joint_learning = specs["JointLearning"]
    num_reg_points = specs["NumRegPoints"]

    # Get the regularization constants for the dynamic regularization
    lambda_motion_lat = specs["LambdaDynRegLat"]
    lambda_motion_l2 = specs["LambdaDynRegL2"]
    lambda_motion_ot = specs["LambdaDynRegOT"]

    # Get the parameters regarding to saving the models and the optimizers
    save_freq_in_epochs = specs['SaveFreq']

    # Get the parameters for the encoder, decoder, and latent warper
    encoder_specs = specs['EncoderSpecs']
    decoder_specs = specs['DecoderSpecs']
    time_warper_specs = specs['TimeWarperSpecs']

    # Define the models
    encoder = Encoder(latent_dim, **encoder_specs)
    decoder = Decoder(latent_dim, **decoder_specs, upsample_size=encoder.output_size)
    # time_warper = NeuralODE(encoder.output_size[0] * encoder.output_size[1] * latent_dim, **time_warper_specs)
    time_warper = NeuralODE(latent_dim, **time_warper_specs)

    # Start up a weights and biases session if we want to use weights and biases
    wandb_dir = os.path.join(experiment_directory, "wandb")
    experiment_name = os.path.basename(experiment_directory)
    if not os.path.isdir(wandb_dir):
        os.makedirs(wandb_dir)
    else:
        if specs["UseWandb"]:
            raise ValueError(
                "Wandb directory already exists. Remove this directory if one wants to rerun the experiment...")
    if specs["UseWandb"]:
        wandb.init(project="WassersteinPrior-based-4dImaging", dir=os.path.join(experiment_directory), config=specs,
                   notes="The experiment directory is: {}".format(experiment_name))
    else:
        wandb.init(project="WassersteinPrior-based-4dImaging", dir=os.path.join(experiment_directory), config=specs,
                   notes="The experiment directory is: {}".format(experiment_name), mode='disabled')

    # Create two datasets: a static one and a dynamic one
    root_dir = specs["DataSource"]
    eval_on = specs["EvalOn"]
    if eval_on not in ['train', 'val']:
        raise ValueError("The option EvalOn should be 'train' or 'eval'")
    static_dataset_tr = dataset_class(
        root_dir, split='train', seed=42, test_size=0.2,
        subsampling=time_subsampling, n_successive=0
    )
    static_dataset_val = dataset_class(
        root_dir, split=eval_on, seed=42, test_size=0.2,
        subsampling=time_subsampling, n_successive=0
    )
    dynamic_dataset_tr = dataset_class(
        root_dir, split='train', seed=42, test_size=0.2,
        subsampling=time_subsampling, n_successive=specs["N"] - 1
    )
    # dynamic_dataset_val = dataset_class(
    #     root_dir, split='val', seed=42, test_size=0.2,
    #     subsampling=time_subsampling, n_successive=specs["N"] - 1
    # )
    # n - 1 since the dataset takes into account the number of successive images from the intitial, so n gives n+1 images
    dynamic_dataset_test = dataset_class(
        root_dir, split=eval_on, seed=42, test_size=0.2,
        full_time_series=True,
    )

    # Create the dataloaders
    train_dataloader_static = DataLoader(
        static_dataset_tr, batch_size=min(len(static_dataset_tr), batch_size_static), shuffle=True, drop_last=True,
        collate_fn=dataset_class.get_collate_fn()
    )
    val_dataloader_static = DataLoader(
        static_dataset_val, batch_size=min(len(static_dataset_val), batch_size_static), shuffle=False, drop_last=False,
        collate_fn=dataset_class.get_collate_fn()
    )
    train_dataloader_dynamic = DataLoader(
        dynamic_dataset_tr, batch_size=min(len(dynamic_dataset_tr), batch_size_dynamic), shuffle=True, drop_last=True,
        collate_fn=dataset_class.get_collate_fn()
    )
    # val_dataloader_dynamic = DataLoader(
    #     dynamic_dataset_val, batch_size=batch_size_dynamic, shuffle=False, drop_last=False,
    #     collate_fn=dataset_class.get_collate_fn()
    # )
    test_dataloader_dynamic = DataLoader(
        dynamic_dataset_test, batch_size=1, shuffle=False, drop_last=False,
        collate_fn=dataset_class.get_collate_fn(),
    )

    # Get the optimizer and the scheduler
    optimizer_all, scheduler = initialize_optimizers_and_schedulers(encoder, decoder, time_warper, init_lr)

    # Put the models on the correct device
    encoder.to(device)
    decoder.to(device)
    time_warper.to(device)

    # In case we want to continue from the latest obtained model, properly initialize the weights of the neural networks
    # and of the optimizers.
    if specs["Continue"] and os.path.isdir(os.path.join(experiment_directory, "ModelParameters")):
        load_model(experiment_directory, encoder, 'encoder', 'static', 'latest.pth')
        load_model(experiment_directory, decoder, 'decoder', 'static', 'latest.pth')
        start_epoch = load_optimizer(experiment_directory, optimizer_all, 'static', 'latest.pth', device)
    else:
        start_epoch = 0

    ######################################################################
    ### First, we only train the encoder and decoder on the image data ###
    ######################################################################

    # For every epoch, do ...
    p_bar = tqdm(range(start_epoch, num_epochs_static), desc="Epoch", total=num_epochs_static, initial=start_epoch)
    metrics_dict = {}
    for epoch in p_bar:

        # Grab a batch
        for imgs, _ in train_dataloader_static:
            # Reset the optimizer
            optimizer_all.zero_grad()

            # Put the batch to the correct device
            imgs = imgs.to(device)

            # Encode
            z, _, _ = encoder(imgs)

            # Decode
            recon = decoder(z)

            # Calculate the reconstruction loss
            loss_recon = recon_loss(recon, imgs)

            # Backpropagate and update the network parameters
            loss_recon.backward()
            optimizer_all.step()

            # Log the reconstruction loss to weights and biases
            metrics_dict.update({'tr/recon': loss_recon.item()})
            p_bar.set_postfix(metrics_dict)

        # Update the learning rate via the scheduler
        scheduler.step()  # assuming the scheduler is stepped per epoch

        if epoch % 5 == 0:
            # do validation
            val_loss = 0
            for imgs, _ in val_dataloader_static:
                imgs = imgs.to(device)
                z, _, _ = encoder(imgs)
                recon = decoder(z)
                val_loss += recon_loss(recon, imgs).item()
            val_loss /= len(val_dataloader_static)
            metrics_dict.update({'val/recon': val_loss})
            p_bar.set_postfix(metrics_dict)

        # Log the training loss
        wandb.log({"loss_recon_static": loss_recon.item()})

        # Save the model
        if epoch % save_freq_in_epochs == 0 or epoch == num_epochs_static - 1:
            save_model_and_optimizer(
                experiment_directory, 'static', epoch, encoder, decoder, time_warper, optimizer_all,
                filename='latest.pth'
            )
            evaluate_random_static_train_reconstructions(
                experiment_directory, encoder, decoder, val_dataloader_static, epoch, device
            )

    ########################################
    ### Now we train everything together ###
    ########################################

    # We reinitialize the optimizers and schedulers
    optimizer_all, scheduler = initialize_optimizers_and_schedulers(encoder, decoder, time_warper, init_lr,
                                                                    joint_learning)

    # In case we want to continue from the latest obtained model, properly initialize the weights of the neural networks
    # and of the optimizers.
    if specs["Continue"] and os.path.isdir(os.path.join(experiment_directory, "ModelParameters")):
        load_model(experiment_directory, encoder, 'encoder', 'dynamic', 'latest.pth')
        load_model(experiment_directory, decoder, 'decoder', 'dynamic', 'latest.pth')
        load_model(experiment_directory, time_warper, 'time_warper', 'dynamic', 'latest.pth')
        start_epoch = load_optimizer(experiment_directory, optimizer_all, 'dynamic', 'latest.pth', device)
    else:
        start_epoch = 0

    # For every epoch, do ...
    p_bar = tqdm(range(start_epoch, num_epochs_dynamic), desc="Epoch", total=num_epochs_dynamic, initial=start_epoch)
    metrics_dict = {}
    for epoch in p_bar:

        # Grab a batch
        for imgs, _ in train_dataloader_dynamic:

            # Reset the optimizer
            optimizer_all.zero_grad()

            # Put the images to the correct device and chunk the image
            imgs = imgs.to(device)
            batch_time_chunked = torch.chunk(imgs, specs["N"], dim=0)

            # The initial images
            img_start = batch_time_chunked[0]
            img_remaining = torch.concat(batch_time_chunked[1:], dim=0)

            # Encode
            z_start, mu_start, log_var_start = encoder(img_start)
            z_remaining, mu_remaining, log_var_remaining = encoder(img_remaining)
            z_static = torch.cat([z_start, z_remaining], dim=0)
            mu_static = None if mu_start is None or mu_remaining is None else torch.cat([mu_start, mu_remaining], dim=0)
            log_var_static = None if log_var_start is None or log_var_remaining is None else torch.cat([log_var_start, log_var_remaining], dim=0)

            # The times at which to evaluate the ode are 0 until the time of the final datapoint in the time series
            # that we have in the batch (note that this is not the full time series but one which contains
            # n_consecutive images).
            num_time_points = specs["N"]

            # The time between two time points of the original time series is nabla_t. Then if we subsample only every
            # time_subsampling points (e.g. if time_subsampling=5, we only sample the images in the time series at
            # indices [0, 5, 10, ...]), we end up at the following end time:
            end_time = nabla_t * time_subsampling * (num_time_points - 1)

            # In case we have an explicit method, we need time steps. We assume num_int_steps to get from 0 to nabla_t.
            if not time_warper.adaptive:
                t_actual = torch.linspace(0.0, end_time, (num_time_points - 1) * num_int_steps * time_subsampling + 1).to(device)
            else:
                t_actual = torch.linspace(0.0, end_time, num_time_points).to(device)

            # Add some random entries to t. Required for calculating regularizers on the time dynamics
            if lambda_motion_lat > 0 or lambda_motion_l2 > 0 or lambda_motion_ot > 0:
                t_rand = torch.rand((num_reg_points,), device=t_actual.device) * (end_time - h - 2 * 1e-6) + (h / 2 + 1e-6)
                t_rand = torch.cat([t_rand - h / 2, t_rand + h / 2])
                t = torch.cat([t_actual, t_rand])
            else:
                t = t_actual
            t, indices = torch.sort(t)

            # Latent dynamics
            z_t = time_warper(z_start, t)

            # One part of the latent codes is used for reconstruction while the other is used for regularization
            z_t_for_reg = z_t[indices >= len(t_actual)]
            z_t_for_recon = z_t[indices < len(t_actual)]

            # We need to subsample in case we use a non-adaptive method
            if not time_warper.adaptive:
                z_t_for_recon = z_t_for_recon[::time_subsampling]

            # z_static at time point i+1 - z_static at time point i
            diff_z_static = torch.diff(torch.stack(torch.chunk(z_static, specs["N"], dim=0), dim=0), dim=0)

            # Loss on whether the static latent codes are relative close and not too large
            latent_regularizer = (0.1 * recon_loss(diff_z_static, torch.zeros_like(diff_z_static)) +
                                  0.02 * recon_loss(z_static, torch.zeros_like(z_static)))

            # Decode everything
            if joint_learning:
                img_recon_static = decoder(z_static)
                img_recon_dynamic = decoder(rearrange(z_t_for_recon, "t b l -> (t b) l"))

                # # Get the reconstruction loss
                if log_var_static is None:
                    loss_recon_static = recon_loss(imgs, img_recon_static)
                    loss_recon_dynamic = recon_loss(imgs[batch_size_dynamic:], img_recon_dynamic[batch_size_dynamic:])
                    latent_regularizer = torch.tensor(0.0).to(device)
                    loss_recon = 2.0 * loss_recon_static + 10.0 * loss_recon_dynamic
                else:
                    vae_loss_static, loss_recon_static, _ = vae_recon_loss(imgs, img_recon_static, mu_static,
                                                                               log_var_static)
                    loss_recon_dynamic = recon_loss(imgs, img_recon_dynamic)
                    latent_regularizer = torch.tensor(0.0).to(device)
                    loss_recon = vae_loss_static + 0.0 * loss_recon_dynamic

            else:
                img_recon_dynamic = decoder(rearrange(z_t_for_recon, "t b l -> (t b) l"))
                loss_recon_static = torch.tensor(0.0, device=device)
                loss_recon_dynamic = recon_loss(imgs[batch_size_dynamic:], img_recon_dynamic[batch_size_dynamic:])
                loss_recon = loss_recon_static + 10 * loss_recon_dynamic

            # Make sure the latent vectors of the static reconstruction are equal to the ones of the dynamic reconstruction
            loss_latent_recon = recon_loss(z_static[batch_size_dynamic:, ...], rearrange(z_t_for_recon, "t b l -> (t b) l")[batch_size_dynamic:, ...])

            if lambda_motion_lat > 0 or lambda_motion_l2 > 0 or lambda_motion_ot > 0:
                # Get reconstructions used for the regularizer
                img_recon_reg = decoder(z_t_for_reg.reshape(-1, latent_dim))

                # Get the left hand points and the right hand points for the regularizer
                z_t_for_recon_left = z_t_for_recon[:num_reg_points]
                z_t_for_recon_right = z_t_for_recon[num_reg_points:]
                img_recon_reg_left = img_recon_reg[:(num_reg_points * batch_size_dynamic), ...]
                img_recon_reg_right = img_recon_reg[(num_reg_points * batch_size_dynamic):, ...]

                # Calculate dynamic regularization
                if lambda_motion_lat > 0:
                    latent_motion_prior = torch.mean(torch.diff(z_t_for_recon_right - z_t_for_recon_left) ** 2)
                else:
                    latent_motion_prior = torch.tensor(0.0, device=device)

                if lambda_motion_l2 > 0:
                    image_motion_prior_l2 = torch.mean((img_recon_reg_right - img_recon_reg_left) ** 2)
                else:
                    image_motion_prior_l2 = torch.tensor(0.0, device=device)

                if lambda_motion_ot > 0:
                    image_motion_prior_ot =  wasserstein_dist(img_recon_reg_right, img_recon_reg_left)
                else:
                    image_motion_prior_ot = torch.tensor(0.0, device=device)
            else:
                latent_motion_prior = torch.tensor(0.0, device=device)
                image_motion_prior_l2 = torch.tensor(0.0, device=device)
                image_motion_prior_ot = torch.tensor(0.0, device=device)

            loss_motion_prior = lambda_motion_lat * latent_motion_prior + lambda_motion_l2 * image_motion_prior_l2 + lambda_motion_ot * image_motion_prior_ot

            # Calculate the full loss
            loss = loss_recon + loss_latent_recon + 0.0 * latent_regularizer + loss_motion_prior

            # Update the parameters
            loss.backward()
            optimizer_all.step()

            # Log the losses to weights and biases
            metrics_dict.update({
                'tr/loss': loss.item(),
                'tr/recon': loss_recon.item(),
                'tr/recon_static': loss_recon_static.item(),
                'tr/recon_dynamic': loss_recon_dynamic.item(),
                'tr/latent_recon': loss_latent_recon.item(),
                'tr/motion_prior': loss_motion_prior.item(),
                'tr/latent_regularizer': latent_regularizer.item()}
            )
            p_bar.set_postfix(metrics_dict)

        # Update the scheduler
        scheduler.step()

        # Log to weights and biases
        wandb.log({
                'tr/loss': loss.item(),
                'tr/recon': loss_recon.item(),
                'tr/recon_static': loss_recon_static.item(),
                'tr/recon_dynamic': loss_recon_dynamic.item(),
                'tr/latent_recon': loss_latent_recon.item(),
                'tr/motion_prior': loss_motion_prior.item(),
                'tr/latent_regularizer': latent_regularizer.item()})

        # Save the model
        if (not epoch == 0) and (epoch % save_freq_in_epochs == 0 or epoch == num_epochs_dynamic - 1):
            save_model_and_optimizer(
                experiment_directory, 'dynamic', epoch, encoder, decoder, time_warper,
                optimizer_all, filename='latest.pth'
            )
            evaluate_random_dynamic_train_reconstructions(
                experiment_directory, encoder, decoder, time_warper,
                test_dataloader_dynamic, nabla_t, num_int_steps,
                time_subsampling, epoch, device,
                p_bar, metrics_dict
            )


if __name__ == "__main__":
    from pathlib import Path

    torch.random.manual_seed(31359)
    np.random.seed(31359)

    arg_parser = argparse.ArgumentParser(description="Train")
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

    args = arg_parser.parse_args()

    train_model(args.experiment_directory)

# img_left_recon = decoder(z_left)
# img_right_recon = decoder(z_right)

##################################################
### Some code for creating a latent space plot ###
##################################################

# # # Reconstruct all of the time points
# # recon_list = []
# # t = torch.linspace(0, 0.5, 11).to('cuda')
# # z_t = z_start[None, ...] * (1 - t[:, None, None]) + t[:, None, None] * z_end[None, ...]
# # z_t = z_dynamic_end[None, ...] * (1 - t[:, None, None]) + t[:, None, None] * z_end[None, ...]
# # for i in range(t.size(0)):
# #     z_i = z_t[i, 3, ...][None, ...]
# #     recon_list.append(decoder(z_i).squeeze().detach().cpu().numpy())
# #     plt.imshow(recon_list[-1])
# #     plt.show()
# #
# # Plotting the latent space
# fig, ax = plt.subplots()
# latent_dataset = dataset_class(
#         root_dir, split=eval_on, seed=42, test_size=0.2, subsampling=time_subsampling,
#         full_time_series=True
#     )
# latent_dataloader = DataLoader(latent_dataset, batch_size=1)
#
# cmap = plt.cm.get_cmap('hsv', len(latent_dataloader))
# for i, (time_series, _) in enumerate(latent_dataloader):
#     if i % 1 == 0:
#         time_series = torch.stack(time_series)
#         print(time_series.size())
#         batch = time_series.squeeze().unsqueeze(1)
#         _, z_thing, _ = encoder(batch.to('cuda'))
#         print(z_thing.size())
#
#         end_time = nabla_t * time_subsampling * (z_thing.size(0) - 1)
#
#         print(end_time)
#
#         t = torch.linspace(0.0, end_time, num_int_steps * time_subsampling * (z_thing.size(0) - 1) + 1).to(device)
#         # z_t = time_warper(z_start.reshape(batch_size, -1), t).reshape(time_subsampling * num_int_steps + 1, -1, latent_dim, encoder.output_size[0], encoder.output_size[1])
#         z_t_thing = time_warper(z_thing[0, ...][None, ...], t)
#         #print(z_t_thing.shape)
#         z_t_thing = torch.cat([z_t_thing[0, ...][None,...], z_t_thing[time_subsampling::time_subsampling]],dim=0)
#         #print(z_t_thing.shape)
#         #print("YEEEEEE")
#         one_step_pred = time_warper(z_thing, torch.linspace(0.0, nabla_t * time_subsampling, num_int_steps * time_subsampling + 1).to(device))
#         #print(one_step_pred.size())
#         #print(z_thing.size())
#         #print("t")
#         one_step_pred = torch.cat([z_thing[0, ...][None, ...], one_step_pred[-1, ...][:-1, ...]], dim=0)
#         #print("t")
#         z_thing = z_thing.detach().cpu().numpy()
#         z_t_thing = z_t_thing.detach().cpu().numpy().squeeze()
#         one_step_pred = one_step_pred.detach().cpu().numpy().squeeze()
#         #print("t")
#         ax.scatter(z_thing[:, 0], z_thing[:, 1], s=60, c=cmap(i))#c='b')#cmap(i))
#         ax.scatter(z_t_thing[:, 0], z_t_thing[:, 1], s=5, c='black', marker='*')#)cmap(i), marker="x")
#         #print(one_step_pred.shape)
#         #ax.scatter(one_step_pred[:, 0], one_step_pred[:, 1], s=5, c='black', marker='x')#c='g')#cmap(len(latent_dataloader)-i-1), marker="*")
#
# plt.show()
# plt.close('all')
#


#
# # Normalize the intermediate images for calculating the motion prior loss
# idx_1, idx_2 = 9, 10
# img_test_1 = recon_list[idx_1] / recon_list[idx_1].sum()
# img_test_2 =recon_list[idx_2] / recon_list[idx_2].sum()
#
# # Calculate the motion prior loss
# wasserstein_dist = FastConvolutionalW2Cost(blur=0.00000005)
# from geomloss import SamplesLoss
#
# xg, yg = np.meshgrid(
#     np.arange(img_test_1.size(0)),
#     np.arange(img_test_1.size(1)),
#     indexing="xy",
# )
#
# t_x = np.linspace(0, 1, 64)
# t_y = np.linspace(0, 1, 80)
# w_s = 64
# h_s = 80
# C_y = (t_x.view(w_s, 1) - (t_x.view(1, w_s))) ** 2
# C_x = (t_y.view(h_s, 1) - (t_y.view(1, h_s))) ** 2
#
# import POT
#
# loss_val = POT.ot.emd2()
#
#
#
# wass_dist = SamplesLoss(loss='sinkhorn', p=2, blur=0.00000005)
# loss_motion_prior = wasserstein_dist(torch.tensor(img_test_1, device=device)[None, None, ...], torch.tensor(img_test_2, device=device)[None, None, ...]) / h


# # Normalize the intermediate images for calculating the motion prior loss
# img_left_recon_normalized = img_left_recon / (img_left_recon.sum(dim=(1, 2, 3))[:, None, None, None])
# img_right_recon_normalized = img_right_recon / (img_right_recon.sum(dim=(1, 2, 3))[:, None, None, None])
#
# # Calculate the motion prior loss
# loss_motion_prior = wasserstein_dist(img_left_recon_normalized, img_right_recon_normalized) / h
