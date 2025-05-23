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
from src.utils.wasserstein_barycenters import convolutional_barycenter_calculation

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

    # Initialize a figure for the latent space
    fig_latent, ax_latent = plt.subplots()
    cmap = plt.cm.get_cmap('hsv', len(dataloader))

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

        # Finally, get a plot of the latent space in case the latent dimension equals 2
        if encoder.latent_dim == 2:
            batch = time_series.squeeze().unsqueeze(1)[::time_subsampling]
            _, z_thing, _ = encoder(batch.to('cuda'))
            end_time = nabla_t * time_subsampling * (z_thing.size(0) - 1)
            t = torch.linspace(0.0, end_time, num_int_steps * time_subsampling * (z_thing.size(0) - 1) + 1).to(device)
            z_t_thing = time_warper(z_thing[0, ...][None, ...], t)
            z_t_thing = torch.cat([z_t_thing[0, ...][None, ...], z_t_thing[time_subsampling::time_subsampling]], dim=0)
            one_step_pred = time_warper(z_thing, torch.linspace(0.0, nabla_t * time_subsampling,
                                                                num_int_steps * time_subsampling + 1).to(device))

            one_step_pred = torch.cat([z_thing[0, ...][None, ...], one_step_pred[-1, ...][:-1, ...]], dim=0)
            z_thing = z_thing.detach().cpu().numpy()
            z_t_thing = z_t_thing.detach().cpu().numpy().squeeze()
            one_step_pred = one_step_pred.detach().cpu().numpy().squeeze()
            ax_latent.scatter(z_thing[:, 0], z_thing[:, 1], s=60, c=cmap(i))
            ax_latent.scatter(z_t_thing[:, 0], z_t_thing[:, 1], s=5, c='b', marker="x")
            ax_latent.scatter(one_step_pred[:, 0], one_step_pred[:, 1], s=5, c=cmap(len(dataloader) - i - 1),
                              marker="*")

    # Save the latent figure
    save_dir = os.path.join(
        experiment_directory, "Figures", "Latent space")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    fig_latent.savefig(os.path.join(save_dir, "{}.png".format(epoch)))

    plt.close('all')

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

    most_common_val = decoder.most_common_val
    if most_common_val > 0.0:
        recon = most_common_val + (1.0 - most_common_val) * recon[:, 0, ...] - most_common_val * recon[:, 1, ...]
        recon = recon[:, None, ...]

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
    nabla_t = specs["Nabla_t"]
    batch_size_static = specs['BatchSizeStatic']
    batch_size_dynamic = specs["BatchSizeDynamic"]
    latent_dim = specs['LatentDim']
    init_lr = specs['InitLR']
    num_int_steps = specs['NumIntSteps']
    time_subsampling = specs["TimeSubsampling"]
    joint_learning = specs["JointLearning"]
    num_reg_points = specs["NumRegPoints"]

    # Get the weighting factor for the dynamic reconstruction loss
    lambda_recon_dynamic = specs["LambdaReconDynamic"]

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
        raise ValueError("The option EvalOn should be 'train' or 'val'")
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
    test_dataloader_dynamic = DataLoader(
        dynamic_dataset_test, batch_size=1, shuffle=False, drop_last=False,
        collate_fn=dataset_class.get_collate_fn(),
    )

    # Save the number of time series per batch
    num_time_series = min(len(dynamic_dataset_tr), batch_size_dynamic)

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
        for xs, _ in train_dataloader_static:
            # Reset the optimizer
            optimizer_all.zero_grad()

            # Put the batch to the correct device
            xs = xs.to(device)

            # Encode
            z, _, _ = encoder(xs)

            # Decode
            recon = decoder(z)

            # Calculate the reconstruction loss
            most_common_val = specs["DecoderSpecs"]['most_common_val']
            if specs["DecoderSpecs"]['most_common_val'] > 0.0:
                target_out_1 = (xs - most_common_val) * (xs >= most_common_val) / (1.0 - most_common_val)
                target_out_2 = (most_common_val - xs) * (xs <= most_common_val) / most_common_val
                loss_recon = 0.5 * (recon_loss(recon[:, 0, ...][:, None, ...], target_out_1) + recon_loss(recon[:, 1, ...][:, None, ...], target_out_2))
            else:
                loss_recon = recon_loss(recon, xs)

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
            for xs, _ in val_dataloader_static:
                xs = xs.to(device)
                z, _, _ = encoder(xs)
                recon = decoder(z)
                if specs["DecoderSpecs"]['most_common_val'] > 0.0:
                    recon = most_common_val + (1.0 - most_common_val) * recon[:, 0, ...] - most_common_val * recon[:, 1,
                                                                                                             ...]
                    recon = recon[:, None, ...]
                val_loss += recon_loss(recon, xs).item()
            val_loss /= len(val_dataloader_static)
            metrics_dict.update({'val/recon': val_loss})
            p_bar.set_postfix(metrics_dict)

        # Log the training loss
        wandb.log({"im_recon_loss": loss_recon.item()})

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
    
    def time_rearrange(_x, _n):
        return torch.stack(torch.chunk(_x, _n, dim=0), dim=0)

    def stack_rearrange(_x):
        return torch.cat(torch.unbind(_x, dim=0), dim=0)

    n = specs["N"]

    # We reinitialize the optimizers and schedulers
    optimizer_all, scheduler = initialize_optimizers_and_schedulers(encoder, decoder, time_warper, init_lr, joint_learning)

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
        for xs, _ in train_dataloader_dynamic:

            # Reset the optimizer
            optimizer_all.zero_grad()

            # Put the images to the correct device and chunk the image
            xs = xs.to(device)
            xt = time_rearrange(xs, n)

            # The initial images
            x0 = xt[0]
            xs_remaining = stack_rearrange(xt[1:])

            # Encode
            z0, mu_start, log_var_start = encoder(x0)
            zs_remaining, mu_remaining, log_var_remaining = encoder(xs_remaining)
            zs_static = torch.cat([z0, zs_remaining], dim=0)

            mu_static = None if mu_start is None or mu_remaining is None else torch.cat([mu_start, mu_remaining], dim=0)
            log_var_static = None if log_var_start is None or log_var_remaining is None else torch.cat(
                [log_var_start, log_var_remaining], dim=0
            )

            # The times at which to evaluate the ode are 0 until the time of the final datapoint in the time series
            # that we have in the batch (note that this is not the full time series but one which contains
            # n_consecutive images).

            # The time between two time points of the original time series is nabla_t. Then if we subsample only every
            # time_subsampling points (e.g. if time_subsampling=5, we only sample the images in the time series at
            # indices [0, 5, 10, ...]), we end up at the following end time:
            end_time = nabla_t * time_subsampling * (n - 1)

            # In case we have an explicit method, we need time steps. We assume num_int_steps to get from 0 to nabla_t.
            if not time_warper.adaptive:
                t_actual = torch.linspace(0.0, end_time, (n - 1) * num_int_steps * time_subsampling + 1).to(device)
            else:
                t_actual = torch.linspace(0.0, end_time, n).to(device)

            # Add some random entries to t. Required for calculating regularizers on the time dynamics
            if lambda_motion_lat > 0 or lambda_motion_l2 > 0 or lambda_motion_ot > 0:
                    
                t_rand = torch.cat([(t_actual[::time_subsampling] + (i+1) * nabla_t * time_subsampling / (num_reg_points+1))[:-1] for i in range(num_reg_points)])
                t_rand, _ = torch.sort(t_rand)
                t = torch.cat([t_actual, t_rand])
                
            else:
                t_rand = None
                t = t_actual
            
            # NEW
            t, indices_inverse = torch.unique(t, return_inverse=True)
            zt = time_warper(z0, t)
            zt_regs = zt[indices_inverse[t_actual.shape[0]:]]
            zt_pred = zt[indices_inverse[:t_actual.shape[0]]]
            
            # We need to subsample in case we use a non-adaptive method
            if not time_warper.adaptive:
                zt_pred = zt_pred[::(num_int_steps * time_subsampling)]

            # zs_static at time point i+1 - zs_static at time point i
            diff_z_static = torch.diff(torch.stack(torch.chunk(zs_static, specs["N"], dim=0), dim=0), dim=0)

            # Loss on whether the static latent codes are relative close and not too large
            latent_regularizer = (
                    0.1 * recon_loss(diff_z_static, torch.zeros_like(diff_z_static)) +
                    0.02 * recon_loss(zs_static, torch.zeros_like(zs_static))
            )

            # Decode everything
            if joint_learning:
                xs_static_preds = decoder(zs_static)
                xs_dynamic_preds = decoder(stack_rearrange(zt_pred))

                # # Get the reconstruction loss
                if log_var_static is None:
                    im_recon_loss = recon_loss(xs, xs_static_preds)
                    dm_loss = recon_loss(xs[num_time_series:], xs_dynamic_preds[num_time_series:])
                    latent_regularizer = torch.tensor(0.0).to(device)
                    loss_recon = im_recon_loss + lambda_recon_dynamic * dm_loss
                else:
                    vae_loss_static, im_recon_loss, _ = vae_recon_loss(xs, xs_static_preds, mu_static, log_var_static)
                    dm_loss = recon_loss(xs[num_time_series:], xs_dynamic_preds[num_time_series:])
                    latent_regularizer = torch.tensor(0.0).to(device)
                    loss_recon = vae_loss_static + lambda_recon_dynamic * dm_loss
            else:
                xs_dynamic_preds = decoder(stack_rearrange(zt_pred))
                im_recon_loss = torch.tensor(0.0, device=device)
                dm_loss = recon_loss(xs[num_time_series:], xs_dynamic_preds[num_time_series:])
                loss_recon = im_recon_loss + lambda_recon_dynamic * dm_loss
            
            # Make sure the latent vectors of the static reconstruction are equal to the ones of the dynamic reconstruction
            loss_latent_recon = recon_loss(zs_static[num_time_series:, ...], stack_rearrange(zt_pred)[num_time_series:, ...])

            # Currently the code only works when either ONLY OT regularization is used OR when latent L2 and image L2 regularization is used
            if lambda_motion_ot > 0:
                if lambda_motion_lat > 0 or lambda_motion_l2 > 0:
                    raise AssertionError("We want to use OT regularization together with ||dz/dt||_2 and/or ||d/dt D(z(t))||_2 regularization." 
                                         +" The current code does not support this yet. Use ONLY OT regularization or use"
                                         +" ||dz/dt||_2 and/or ||d/dt D(z(t))||_2 regularization...")
            
            # Calculate the regularization terms
            if lambda_motion_ot > 0:
                
                # Get the reconstruction
                img_recon_reg = decoder(stack_rearrange(zt_regs))
                
                # Then for each of the random numbers, we look at the interval it belongs to
                intval_idx = torch.floor(t_rand / (nabla_t * time_subsampling)).int()

                # Get the points where we have data and that correspond to the regularization ones
                barycenter_boundaries = time_rearrange(xs_dynamic_preds, n)
                bary_center_boundaries_normalization_factor = torch.sum(barycenter_boundaries, dim=(2, 3, 4), keepdim=True)
                
                barycenter_boundaries = barycenter_boundaries / bary_center_boundaries_normalization_factor
                barycenter_boundary_left, barycenter_boundary_right = barycenter_boundaries[intval_idx], \
                barycenter_boundaries[intval_idx + 1]
                
                # Get pairs of weights depending on the sampled random time points
                img_pairs = torch.stack([barycenter_boundary_left, barycenter_boundary_right], dim=0)
                weights_left = 1.0 - torch.remainder(t_rand, (nabla_t * time_subsampling)) / (
                            nabla_t * time_subsampling)
                weights_right = 1.0 - weights_left
                weights = torch.stack([weights_left, weights_right], dim=0)[..., None, None, None, None]
                weights = weights.expand(-1, -1, img_pairs.size(2), -1, -1, -1)
                img_pairs = rearrange(img_pairs, 'm t b c h w -> m (t b) c h w')
                weights = rearrange(weights, 'm t b c h w -> m (t b) c h w')

                # Calculate the barycenters
                barycenters = convolutional_barycenter_calculation(img_pairs, weights=weights, scaling=0.95, need_diffable=True)

                # Calculate the loss
                scaling = (bary_center_boundaries_normalization_factor[intval_idx, ...] * weights_left[:, None,
                                                                                            None, None, None] +
                            bary_center_boundaries_normalization_factor[intval_idx + 1, ...] * weights_right[:, None,
                                                                                                None, None, None])
                scaling = rearrange(scaling, 't b c h w -> (t b) c h w')
                img_recon_reg_scaled = img_recon_reg
                barycenters = (barycenters / barycenters.sum(dim=(1, 2, 3))[:, None, None, None]) * scaling
                barycenter_loss = recon_loss(img_recon_reg_scaled, barycenters)
                image_motion_prior_ot = barycenter_loss

                # Put the other motion priors to zero
                latent_motion_prior = torch.tensor(0.0, device=device)
                image_motion_prior_l2 = torch.tensor(0.0, device=device)
            elif lambda_motion_lat > 0 or lambda_motion_l2 > 0 or lambda_motion_ot > 0: 
                    
                img_recon_reg = time_rearrange(decoder(stack_rearrange(zt_regs)), t_rand.shape[0])
                
                # Then for each of the random numbers, we look at the interval it belongs to
                intval_idx = torch.floor(t_rand / (nabla_t * time_subsampling)).int()

                # Get the points where we have data and that correspond to the regularization ones
                barycenter_boundaries = time_rearrange(xs_dynamic_preds, n)
                zt_boundaries = zt_pred

                # get the full intervals
                zt_regs_full = torch.stack([torch.cat([zt_boundaries[i, ...].unsqueeze(0), zt_regs[intval_idx==i], zt_boundaries[i+1, ...].unsqueeze(0)], dim=0) 
                                            for i in range(n-1)], dim=0)
                xt_regs_full = torch.stack([torch.cat([barycenter_boundaries[i, ...].unsqueeze(0), img_recon_reg[intval_idx==i], barycenter_boundaries[i+1, ...].unsqueeze(0)], dim=0) 
                                            for i in range(n-1)], dim=0)
                
                # For each interval
                l2_prior = torch.nn.MSELoss(reduction='none')
                if lambda_motion_lat > 0:
                    path_length = torch.sum(
                        torch.stack([
                            l2_prior(zt_regs_full[:, i, ...], zt_regs_full[:, i + 1, ...]) for i in range(num_reg_points + 1)
                        ], dim=0), dim=0
                    )
                    latent_motion_prior = torch.mean(path_length)
                else:
                    latent_motion_prior = torch.tensor(0.0, device=device)

                if lambda_motion_l2 > 0:
                    image_motion_prior_l2 = torch.sum(
                        torch.stack([
                            l2_prior(xt_regs_full[:, i, ...], xt_regs_full[:, i + 1, ...]) for i in range(num_reg_points + 1)
                        ], dim=0), dim=0
                    )
                    image_motion_prior_l2 = torch.mean(image_motion_prior_l2)
                else:
                    image_motion_prior_l2 = torch.tensor(0.0, device=device)
                
                image_motion_prior_ot = torch.tensor(0.0, device=device)
                barycenter_loss = torch.tensor(0.0, device=device)
                    
            else:
                latent_motion_prior = torch.tensor(0.0, device=device)
                image_motion_prior_l2 = torch.tensor(0.0, device=device)
                barycenter_loss = torch.tensor(0.0, device=device)
                image_motion_prior_ot = barycenter_loss

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
                'tr/recon_static': im_recon_loss.item(),
                'tr/recon_dynamic': dm_loss.item(),
                'tr/latent_recon': loss_latent_recon.item(),
                'tr/motion_prior': loss_motion_prior.item(),
                'tr/latent_regularizer': latent_regularizer.item(),
                'tr/image_motion_prior_ot': image_motion_prior_ot.item(),
                'tr/barycenter_loss': barycenter_loss.item(),
                'tr/latent_motion_prior': latent_motion_prior.item(),
                'tr/image_motion_prior_l2': image_motion_prior_l2.item()
            })

            p_bar.set_postfix(metrics_dict)

        # Update the scheduler
        scheduler.step()

        # Log to weights and biases
        wandb.log({
            'tr/loss': loss.item(),
            'tr/recon': loss_recon.item(),
            'tr/recon_static': im_recon_loss.item(),
            'tr/recon_dynamic': dm_loss.item(),
            'tr/latent_recon': loss_latent_recon.item(),
            'tr/motion_prior': loss_motion_prior.item(),
            'tr/latent_regularizer': latent_regularizer.item(),
            'tr/image_motion_prior_ot': image_motion_prior_ot.item(),
            'tr/barycenter_loss': barycenter_loss.item()
        })

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
