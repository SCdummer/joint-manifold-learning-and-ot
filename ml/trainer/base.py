import os

from pathlib import Path

import torch
import torchmetrics
import lightning as pl
import matplotlib.pyplot as plt
import wandb

from ml.optim import init_optims_from_config


class BaseTrainer(pl.LightningModule):
    def __init__(self, config, model, loss_fn, metrics, normalisation, de_normalisation):
        super().__init__()

        self.config = config

        # initialise the model
        self.model = model

        # set the loss function
        self.loss_fn = loss_fn

        # set the normalisation function
        self.normalisation = normalisation

        # set the de-normalisation function
        self.de_normalisation = de_normalisation

        metrics = torchmetrics.MetricCollection(metrics)
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

        if wandb.run:
            _id = wandb.run.id
        else:
            import uuid
            _id = uuid.uuid4().hex
        self.dir_name = Path('out', f'{_id}', 'val_images')
        os.makedirs(self.dir_name, exist_ok=True)

    def forward(self, x):
        return self.model(self.normalisation(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        # concatenate the input and target images on the batch dimension
        _stacked = torch.cat((x, y), dim=0)
        _normalised_stacked = self.normalisation(_stacked)
        _stacked_pred, _ = self.model(_normalised_stacked)

        loss = self.loss_fn(_stacked_pred, _normalised_stacked)
        self.log('train/loss', loss, prog_bar=True, on_epoch=False, on_step=True)
        self.train_metrics.update(_stacked_pred, _normalised_stacked)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _stacked = torch.cat((x, y), dim=0)
        _normalised_stacked = self.normalisation(_stacked)
        _stacked_pred, _ = self.model(_normalised_stacked)
        loss = self.loss_fn(_stacked_pred, _normalised_stacked)
        self.log('val/loss', loss, prog_bar=False, on_epoch=True, on_step=False)
        self.val_metrics.update(_stacked_pred, _normalised_stacked)
        if batch_idx % 5 == 0:
            # save the images from the first batch
            y_hat = self.de_normalisation(_stacked_pred)
            np_x, np_y_hat = x.detach().cpu().numpy(), y_hat.detach().cpu().numpy()
            _, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(np_x[0].squeeze(), cmap='gray')
            axs[1].imshow(np_y_hat[0].squeeze(), cmap='gray')

            path_to_save_to = Path(self.dir_name, f'recon_val', f'{batch_idx}', f'epoch={self.current_epoch}.png')
            os.makedirs(path_to_save_to.parent, exist_ok=True)
            plt.savefig(path_to_save_to)
            plt.close()
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        _stacked = torch.cat((x, y), dim=0)
        _normalised_stacked = self.normalisation(_stacked)
        _stacked_pred, _ = self.model(_normalised_stacked)
        loss = self.loss_fn(_stacked_pred, _normalised_stacked)
        self.log('test/loss', loss, prog_bar=False, on_epoch=True, on_step=False)
        self.test_metrics.update(_stacked_pred, _normalised_stacked)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), prog_bar=False, on_epoch=True, on_step=False)
        self.test_metrics.reset()

    def configure_optimizers(self):
        # initialise optimizers and lr schedulers
        return init_optims_from_config(self.config, self.model)


class ODETrainer(BaseTrainer):
    def __init__(
            self,
            config, model, loss_fn, metrics, normalisation, de_normalisation,
            encoder, decoder
    ):
        super().__init__(config, model, loss_fn, metrics, normalisation, de_normalisation)
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()

        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)

    def forward(self, x):
        return self.decoder(self.model(self.encoder(self.normalisation(x))))

    def training_step(self, batch, batch_idx):
        x0, x1 = batch
        _stacked = torch.cat((x0, x1), dim=0)
        _normalised_stacked = self.normalisation(_stacked)
        _stacked_latents = self.encoder(_normalised_stacked)
        z0, z1 = torch.chunk(_stacked_latents, 2, dim=0)
        zt = self.model(z0)  # (t, b, c, 1, 1) where t is the number of time steps

        # for now, we will only consider the last time step
        zt = zt[-1]  # (b, c, 1, 1)

        loss = self.loss_fn(zt, z1)
        self.log('train/loss', loss, prog_bar=True, on_epoch=False, on_step=True)
        self.train_metrics.update(zt, z1)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, x1 = batch
        _stacked = torch.cat((x0, x1), dim=0)
        _normalised_stacked = self.normalisation(_stacked)
        _stacked_latents = self.encoder(_normalised_stacked)
        z0, z1 = torch.chunk(_stacked_latents, 2, dim=0)
        zt = self.model(z0)

        zt = zt[-1]

        latent_loss = self.loss_fn(zt, z1)
        y1 = self.decoder(zt)
        reconstruction_loss = self.loss_fn(y1, x1)
        loss = latent_loss + reconstruction_loss
        self.log('val/loss', loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val/l_loss', latent_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val/r_loss', reconstruction_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.val_metrics.update(zt, z1)
        if batch_idx % 5 == 0:
            # save the images from the first batch
            y_hat = self.de_normalisation(y1)
            np_x0, np_y_hat, np_x1 = x1.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), x1.detach().cpu().numpy()
            _, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(np_x0[0].squeeze(), cmap='gray')
            axs[1].imshow(np_y_hat[0].squeeze(), cmap='gray')
            axs[2].imshow(np_x1[0].squeeze(), cmap='gray')

            path_to_save_to = Path(self.dir_name, f'ode_val', f'{batch_idx}', f'epoch={self.current_epoch}.png')
            os.makedirs(path_to_save_to.parent, exist_ok=True)
            plt.savefig(path_to_save_to)
            plt.close()

        return loss


class JointReconODETrainer(ODETrainer):
    def __init__(
            self,
            config, model, loss_fn, metrics, normalisation, de_normalisation,
            encoder, decoder
    ):
        super().__init__(config, model, loss_fn, metrics, normalisation, de_normalisation, encoder, decoder)
        self.encoder = encoder.train()
        self.decoder = decoder.train()

        self.encoder.requires_grad_(True)
        self.decoder.requires_grad_(True)

    def training_step(self, batch, batch_idx):
        x0, x1 = batch
        _stacked = torch.cat((x0, x1), dim=0)
        _normalised_stacked = self.normalisation(_stacked)
        _stacked_latents = self.encoder(_normalised_stacked)
        z0, z1 = torch.chunk(_stacked_latents, 2, dim=0)
        zt = self.model(z0)

        # for now, we will only consider the last time step
        zt = zt[-1]

        l_loss = self.loss_fn(zt, z1)

        _stacked_latents = torch.cat((z0, z1, zt), dim=0)
        _stacked_recon = self.decoder(_stacked_latents)

        y0, y1, yt = torch.chunk(_stacked_recon, 3, dim=0)
        recon_loss = (self.loss_fn(y0, x0) + self.loss_fn(y1, x1)) / 2
        data_match_loss = self.loss_fn(yt, x1)

        loss = (recon_loss + l_loss + data_match_loss) / 3

        self.log('train/loss', loss, prog_bar=True, on_epoch=False, on_step=True)
        self.log('train/l_loss', l_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train/r_loss', recon_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train/dm_loss', data_match_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.train_metrics.update(yt, x1)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, x1 = batch
        _stacked = torch.cat((x0, x1), dim=0)
        _normalised_stacked = self.normalisation(_stacked)
        _stacked_latents = self.encoder(_normalised_stacked)
        z0, z1 = torch.chunk(_stacked_latents, 2, dim=0)
        zt = self.model(z0)

        zt = zt[-1]

        latent_loss = self.loss_fn(zt, z1)
        _stacked_latents = torch.cat((z0, z1, zt), dim=0)
        _stacked_recon = self.decoder(_stacked_latents)

        y0, y1, yt = torch.chunk(_stacked_recon, 3, dim=0)
        reconstruction_loss = (self.loss_fn(y0, x0) + self.loss_fn(y1, x1)) / 2
        data_match_loss = self.loss_fn(yt, x1)

        loss = (reconstruction_loss + latent_loss + data_match_loss) / 3
        self.log('val/loss', loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val/l_loss', latent_loss, prog_bar=False, on_epoch=True, on_step=False)
        self.log('val/r_loss', reconstruction_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val/dm_loss', data_match_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.val_metrics.update(yt, x1)
        if batch_idx % 5 == 0:
            # save the images from the first batch
            y_hat = self.de_normalisation(yt)
            np_x0, np_y_hat, np_x1 = x1.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), x1.detach().cpu().numpy()
            _, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(np_x0[0].squeeze(), cmap='gray')
            axs[1].imshow(np_y_hat[0].squeeze(), cmap='gray')
            axs[2].imshow(np_x1[0].squeeze(), cmap='gray')

            path_to_save_to = Path(self.dir_name, f'joint_recon_ode_val', f'{batch_idx}', f'epoch={self.current_epoch}.png')
            os.makedirs(path_to_save_to.parent, exist_ok=True)
            plt.savefig(path_to_save_to)
            plt.close()

        return loss

    def configure_optimizers(self):
        # initialise optimizers and lr schedulers
        optim = init_optims_from_config(self.config, self.model, self.encoder, self.decoder)
        return optim
