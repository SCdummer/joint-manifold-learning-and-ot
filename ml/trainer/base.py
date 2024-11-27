import os

from pathlib import Path

import torch
import torch.nn.functional as F
import torchmetrics
import lightning as pl
import matplotlib.pyplot as plt

from ml.optim import init_optims_from_config, get_regulariser


class MSSSIML1Loss(torch.nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(
            self,
            gaussian_sigmas=(0.5, 1.0, 2.0, 4.0, 8.0),
            data_range=1.0,
            k=(0.01, 0.03),
            alpha=0.025,
    ):
        super(MSSSIML1Loss, self).__init__()
        self.DR = data_range
        self.C1 = (k[0] * data_range) ** 2
        self.C2 = (k[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # c0,c1,...,cM
            g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)

        self.g_masks = torch.nn.Parameter(g_masks, requires_grad=False)

    @staticmethod
    def _fspecial_gauss_1d(size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    @staticmethod
    def _fspecial_gauss_2d(size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = MSSSIML1Loss._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        mux = F.conv2d(x, self.g_masks, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 5, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]

        loss_l1 = F.huber_loss(x, y, reduction='none')  # [B, 1, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(
            loss_l1, self.g_masks.narrow(dim=0, start=-1, length=1), padding=self.pad
        ).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR

        return loss_mix.mean()


class KoleoLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.pdist = torch.nn.PairwiseDistance(p=2)

    def forward(self, latents):
        # latents (b, c, 1, 1)
        vector = latents.squeeze(-1).squeeze(-1)

        # we want to penalise the latents that are close to each other
        # take the dot product of the latents
        dot_product = torch.mm(vector, vector.t())  # (b, b)
        dot_product.view(-1)[::(latents.size(0) + 1)].fill_(-1)  # fill the diagonal with -1
        # max inner product of the latents -> min distance between the latents
        # and we want to penalise the latents that are close to each other so we choose the max indices
        _, indices = torch.max(dot_product, dim=1)
        # take the log of the max dot product
        koleo = - torch.log(self.pdist(vector, vector[indices]) * latents.size(0))
        koleo = torch.where(koleo < 0, torch.zeros_like(koleo), koleo)  # all vectors might already be far apart
        return koleo.mean()


class JointReconODETrainer(pl.LightningModule):

    def __init__(
            self,
            config, normalisation, de_normalisation, metrics, *, neural_ode, encoder, decoder
    ):
        super().__init__()

        self.config = config
        self.n = self.config.dataset.train.n_successive + 1

        # initialise the model
        self.neural_ode = neural_ode
        self.encoder = encoder
        self.decoder = decoder

        # set the loss function
        # self.recon = MSSSIML1Loss(gaussian_sigmas=(0.5, 1.0, 2.0))
        self.recon = torch.nn.HuberLoss()
        self.mse = torch.nn.MSELoss()
        self.koleo = KoleoLoss()
        self.ot_regulariser = get_regulariser(
            'ot', scaling= 0.7, use_pykeops=True, n_per_eps=1, n_extra_last_scale=14
        )

        # set the normalisation function
        self.normalisation = normalisation

        # set the de-normalisation function
        self.de_normalisation = de_normalisation

        metrics = torchmetrics.MetricCollection(metrics)
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

        self.dir_name = Path(self.config.logging_path, 'val_images')
        os.makedirs(self.dir_name, exist_ok=True)

    def metric_log(self, prefix, loss, l_loss, r_loss, dm_loss, s_loss, ot_loss, x_pred, x_gt):
        self.log(f'{prefix}/loss', loss, prog_bar=True, on_epoch=prefix == 'val', on_step=prefix == 'train')
        self.log(f'{prefix}/rec', r_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{prefix}/dm', dm_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{prefix}/ot', ot_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{prefix}/tevol', l_loss, prog_bar=True, on_epoch=True, on_step=False)
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

        xt = self.normalisation(xt)
        zt = self.encoder(xt)  # (b, c, 1, 1)
        _koleo = self.koleo(zt)  # koleo regulariser for spreading the latents
        xt_pred = self.decoder(zt)  # (b * n, c, h, w)
        recon_loss = self.recon(xt, xt_pred)  # reconstruction loss

        ot_loss = []
        smoothness_loss = 0.
        last_z = torch.chunk(zt, self.n, dim=0)[0]
        zt_end_preds = []
        for _ in range(1, self.n):
            z_evolution, z_at_rand_step = self.neural_ode(last_z)  # (time_steps, b, c, 1, 1), (2, b, c, 1, 1)
            last_z = z_evolution[-1]
            zt_end_preds.append(last_z)
            diff = torch.diff(z_evolution, dim=0)
            smoothness_loss += self.mse(
                torch.sum(diff ** 2, dim=(0, -1, -2, -3)),
                torch.sum((z_evolution[-1] - last_z) ** 2, dim=(1, 2, 3))
            )
            x_at_rand_minus, x_at_rand_plus = self.decoder(z_at_rand_step[0]), self.decoder(z_at_rand_step[1])
            x_at_rand_minus = torch.clamp(self.de_normalisation(x_at_rand_minus), 0, 1)
            x_at_rand_plus = torch.clamp(self.de_normalisation(x_at_rand_plus), 0, 1)
            ot_loss.append(self.ot_regulariser(x_at_rand_minus, x_at_rand_plus) / 0.01)

        smoothness_loss = smoothness_loss / (self.n - 1)
        ot_loss = torch.mean(torch.stack(ot_loss))

        zt_end_preds = torch.concat(zt_end_preds, dim=0)  # (b * (n - 1), c, 1, 1)
        xt_end_preds = self.decoder(zt_end_preds)  # (b * (n - 1), c, h, w)

        zt_ends = torch.concat(torch.chunk(zt, self.n, dim=0)[1:], dim=0)
        l_loss = self.mse(zt_ends, zt_end_preds)  # the predicted latents should be close to the actual latents

        # reconstruction loss for the next time step
        xt_end = torch.concat(torch.chunk(xt, self.n, dim=0)[1:], dim=0)
        data_match_loss = self.recon(xt_end, xt_end_preds)  # the predicted images should be close to the actual images

        loss = (recon_loss + 0.1 * _koleo) + (data_match_loss + 0.1 * ot_loss) + 10 * l_loss

        self.metric_log(
            prefix='train',
            loss=loss, l_loss=l_loss, r_loss=recon_loss, dm_loss=data_match_loss, s_loss=smoothness_loss, ot_loss=ot_loss,
            x_pred=xt_end_preds, x_gt=xt_end
        )

        return loss

    def validation_step(self, batch, batch_idx):
        xt, _ = batch  # xt (b, c, h, w), t (b,)

        xt = self.normalisation(xt)
        zt = self.encoder(xt)  # (b, c, 1, 1)
        _koleo = self.koleo(zt)  # koleo regulariser for spreading the latents
        xt_pred = self.decoder(zt)  # (b * n, c, h, w)
        recon_loss = self.recon(xt, xt_pred)  # reconstruction loss

        ot_loss = []
        smoothness_loss = 0.
        last_z = torch.chunk(zt, self.n, dim=0)[0]
        zt_end_preds = []
        for _ in range(1, self.n):
            z_evolution, z_at_rand_step = self.neural_ode(last_z)  # (time_steps, b, c, 1, 1), (2, b, c, 1, 1)
            last_z = z_evolution[-1]
            zt_end_preds.append(last_z)
            diff = torch.diff(z_evolution, dim=0)
            smoothness_loss += self.mse(
                torch.sum(diff ** 2, dim=(0, -1, -2, -3)),
                torch.sum((z_evolution[-1] - last_z) ** 2, dim=(1, 2, 3))
            )
            x_at_rand_minus, x_at_rand_plus = self.decoder(z_at_rand_step[0]), self.decoder(z_at_rand_step[1])
            x_at_rand_minus = torch.clamp(self.de_normalisation(x_at_rand_minus), 0, 1)
            x_at_rand_plus = torch.clamp(self.de_normalisation(x_at_rand_plus), 0, 1)
            ot_loss.append(self.ot_regulariser(x_at_rand_minus, x_at_rand_plus) / 0.01)

        smoothness_loss = smoothness_loss / (self.n - 1)
        ot_loss = torch.mean(torch.stack(ot_loss))

        zt_end_preds = torch.concat(zt_end_preds, dim=0)  # (b * (n - 1), c, 1, 1)
        xt_end_preds = self.decoder(zt_end_preds)  # (b * (n - 1), c, h, w)

        zt_ends = torch.concat(torch.chunk(zt, self.n, dim=0)[1:], dim=0)
        l_loss = self.mse(zt_ends, zt_end_preds)  # the predicted latents should be close to the actual latents

        # reconstruction loss for the next time step
        xt_end = torch.concat(torch.chunk(xt, self.n, dim=0)[1:], dim=0)
        data_match_loss = self.recon(xt_end, xt_end_preds)  # the predicted images should be close to the actual images

        loss = (recon_loss + 0.1 * _koleo) + (data_match_loss + 0.1 * smoothness_loss) + 10 * l_loss

        self.metric_log(
            prefix='val',
            loss=loss, l_loss=l_loss, r_loss=recon_loss, dm_loss=data_match_loss, s_loss=smoothness_loss, ot_loss=ot_loss,
            x_pred=xt_end_preds, x_gt=xt_end
        )

        if batch_idx % 5 == 0:
            # save the images from the first batch
            skipper = xt_end.size(0) // (self.n - 1)  # we removed one chunk
            dn_x, dn_xpred = self.de_normalisation(xt_end)[::skipper], self.de_normalisation(xt_end_preds)[::skipper]
            np_x, np_xpred = dn_x.detach().cpu().numpy(), dn_xpred.detach().cpu().numpy()
            _, axs = plt.subplots(2, len(np_x), figsize=(4 * len(np_x), 12))
            for i in range(len(np_x)):
                # do not show the grid for any of the images
                for ax in axs[:, i]:
                    ax.axis('off')

                axs[0, i].imshow(np_x[i].squeeze(), cmap='gray')
                axs[1, i].imshow(np_xpred[i].squeeze(), cmap='gray')

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.tight_layout()
            path_to_save_to = Path(
                self.dir_name, f'joint_recon_ode_val', f'{batch_idx}', f'epoch={self.current_epoch}.png'
            )
            os.makedirs(path_to_save_to.parent, exist_ok=True)
            plt.savefig(path_to_save_to)
            plt.close()

        return loss

    def configure_optimizers(self):
        # initialise optimizers and lr schedulers
        optim = init_optims_from_config(self.config, self.neural_ode, self.encoder, self.decoder)
        return optim
