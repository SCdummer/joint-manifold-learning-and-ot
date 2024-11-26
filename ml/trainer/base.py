import os

from pathlib import Path

import wandb
import torch
import torchmetrics
import lightning as pl
import matplotlib.pyplot as plt

from einops import rearrange, parse_shape
from torch.nn import PairwiseDistance

from ml.optim import init_optims_from_config


def _koleo_loss(latents):
    pdist = PairwiseDistance(p=2)
    # latents (b, c, 1, 1)
    # we want to penalise the latents that are close to each other
    # we will use the koleo loss to do this that takes the log of the euclidean distance between the latents
    # and then squares it
    vector = latents.squeeze(-1).squeeze(-1)
    # project the latents to a unit sphere
    # vector = vector / torch.norm(vector, dim=1, keepdim=True, p=2)
    # take the dot product of the latents
    dot_product = torch.mm(vector, vector.t())
    dot_product.view(-1)[::(latents.size(0) + 1)].fill_(-1)
    # max inner product of the latents -> min distance between the latents
    # we want to penalise the latents that are close to each other
    _, indices = torch.max(dot_product, dim=1)
    # take the log of the max dot product
    koleo = - torch.log(pdist(vector, vector[indices]) * latents.size(0))
    loss = torch.where(koleo < 0, torch.zeros_like(koleo), koleo)
    return loss.mean()


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
        _stacked_pred, _stacked_latents = self.model(_normalised_stacked)

        koleo_loss = _koleo_loss(_stacked_latents)
        recon_loss = self.loss_fn(_stacked_pred, _normalised_stacked)
        loss = recon_loss + 0.01 * koleo_loss

        self.log('train/loss', recon_loss, prog_bar=True, on_epoch=False, on_step=True)
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

        # sample a random time step except the last one
        # t_rand = torch.randint(1, zt.size(0) - 1, (1,)).item()
        # t = torch.tensor([t_rand / (zt.size(0) - 1)], device=zt.device)

        # zt_rand = zt[t_rand]

        zt_end = zt[-1]  # (b, c, 1, 1)

        # ut = z1 - z0  # actual velocity for latent change
        # vt = zt_end - z0  # predicted velocity for latent change
        # vt_rand = zt_rand - z0  # predicted velocity for latent change at random time step

        t_loss = self.loss_fn(zt_end, z1)

        loss = t_loss
        self.log('train/loss', loss, prog_bar=True, on_epoch=False, on_step=True)
        self.train_metrics.update(zt_end, z1)
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
        if batch_idx % 2 == 0:
            # save the images from the first batch
            y_hat = self.de_normalisation(y1)
            np_x0, np_y_hat, np_x1 = x0.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), x1.detach().cpu().numpy()
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
            encoder, decoder, n=4
    ):
        super().__init__(config, model, loss_fn, metrics, normalisation, de_normalisation, encoder, decoder)
        self.encoder = encoder.train()
        self.decoder = decoder.train()
        self.encoder.requires_grad_(True)
        self.decoder.requires_grad_(True)
        self.n = n

    def metric_log(self, prefix, loss, l_loss, r_loss, dm_loss, s_loss, x_pred, x_gt):
        self.log(f'{prefix}/loss', loss, prog_bar=True, on_epoch=prefix == 'val', on_step=prefix == 'train')
        self.log(f'{prefix}/tevol', l_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{prefix}/rec', r_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{prefix}/dm', dm_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{prefix}/sm', s_loss, prog_bar=True, on_epoch=True, on_step=False)
        if prefix == 'train':
            metrics = self.train_metrics
        elif prefix == 'val':
            metrics = self.val_metrics
        else:
            metrics = self.test_metrics
        metrics.update(x_pred, x_gt)

    def training_step(self, batch, batch_idx):
        xt, _ = batch  # xt (b, c, h, w), t (b,)

        xt = self.normalisation(xt)  # normalise the images
        zt = self.encoder(xt)  # (b, c, 1, 1) low dimensional manifold representation
        _koleo = _koleo_loss(zt)  # koleo regulariser for spreading the latents
        xt_pred = self.decoder(zt)  # (b * n, c, h, w)
        recon_loss = self.loss_fn(xt, xt_pred)  # reconstruction loss

        zt_prediction = self.model(zt)  # predict latents at next time step
        # (time_steps, b * n, c, 1, 1)
        diff = torch.diff(zt, dim=0)
        smoothness_loss = self.loss_fn(diff, torch.zeros_like(diff))

        zt_prediction = zt_prediction[-1]  # (b * n, c, 1, 1)
        xt_1_pred = self.decoder(zt_prediction)  # (b * n, c, h, w)

        zt_ends = torch.concat(torch.chunk(zt, self.n, dim=0)[1:], dim=0)
        zt_1_ends = torch.concat(torch.chunk(zt_prediction, self.n, dim=0)[:-1], dim=0)
        l_loss = self.loss_fn(zt_ends, zt_1_ends)  # matching at latent loss

        # reconstruction loss for the next time step
        xt_end = torch.concat(torch.chunk(xt, self.n, dim=0)[1:], dim=0)
        xt_1_pred = torch.concat(torch.chunk(xt_1_pred, self.n, dim=0)[:-1], dim=0)
        data_match_loss = self.loss_fn(xt_end, xt_1_pred)

        loss = 0.4 * (recon_loss + 0.1 * _koleo) + 0.4 * l_loss + 0.1 * data_match_loss + 0.1 * smoothness_loss

        self.metric_log(
            prefix='train',
            loss=loss, l_loss=l_loss, r_loss=recon_loss, dm_loss=data_match_loss, s_loss=smoothness_loss,
            x_pred=xt_1_pred, x_gt=xt_end
        )

        return loss

    def validation_step(self, batch, batch_idx):
        xt, _ = batch  # xt (b, c, h, w), t (b,)

        xt = self.normalisation(xt)
        zt = self.encoder(xt)  # (b, c, 1, 1)
        _koleo = _koleo_loss(zt)  # koleo regulariser for spreading the latents
        xt_pred = self.decoder(zt)  # (b * n, c, h, w)
        recon_loss = self.loss_fn(xt, xt_pred)  # reconstruction loss

        zt_prediction = self.model(zt)  # predict latents at next time step
        # (time_steps, b * n, c, 1, 1)
        diff = torch.diff(zt, dim=0)
        smoothness_loss = self.loss_fn(diff, torch.zeros_like(diff))

        zt_prediction = zt_prediction[-1]  # (b * n, c, 1, 1)
        xt_1_pred = self.decoder(zt_prediction)  # (b * n, c, h, w)

        zt_ends = torch.concat(torch.chunk(zt, self.n, dim=0)[1:], dim=0)
        zt_1_ends = torch.concat(torch.chunk(zt_prediction, self.n, dim=0)[:-1], dim=0)
        l_loss = self.loss_fn(zt_ends, zt_1_ends)  # matching at latent loss

        # reconstruction loss for the next time step
        xt_end = torch.concat(torch.chunk(xt, self.n, dim=0)[1:], dim=0)
        xt_1_pred = torch.concat(torch.chunk(xt_1_pred, self.n, dim=0)[:-1], dim=0)
        data_match_loss = self.loss_fn(xt_end, xt_1_pred)

        loss = 0.4 * (recon_loss + 0.1 * _koleo) + 0.4 * l_loss + 0.1 * data_match_loss + 0.1 * smoothness_loss

        self.metric_log(
            prefix='val',
            loss=loss, l_loss=l_loss, r_loss=recon_loss, dm_loss=data_match_loss, s_loss=smoothness_loss,
            x_pred=xt_1_pred, x_gt=xt_end
        )

        if batch_idx % 5 == 0:
            # save the images from the first batch
            skipper = xt.size(0) // self.n
            _x0, _xpred = self.de_normalisation(xt)[::skipper], self.de_normalisation(xt_pred)[::skipper]
            np_x0, np_xpred = _x0.detach().cpu().numpy(), _xpred.detach().cpu().numpy()
            _, axs = plt.subplots(2, len(np_x0), figsize=(4 * len(np_x0), 12))
            for i in range(len(np_x0)):
                # do not show the grid for any of the images
                for ax in axs[:, i]:
                    ax.axis('off')

                axs[0, i].imshow(np_x0[i].squeeze(), cmap='gray')
                axs[1, i].imshow(np_xpred[i].squeeze(), cmap='gray')

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.tight_layout()
            path_to_save_to = Path(self.dir_name, f'joint_recon_ode_val', f'{batch_idx}', f'epoch={self.current_epoch}.png')
            os.makedirs(path_to_save_to.parent, exist_ok=True)
            plt.savefig(path_to_save_to)
            plt.close()

        return loss

    def configure_optimizers(self):
        # initialise optimizers and lr schedulers
        optim = init_optims_from_config(self.config, self.model, self.encoder, self.decoder)
        return optim
