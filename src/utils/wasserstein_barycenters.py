import math

import torch
import numpy as np

from einops import rearrange, parse_shape
from torch.nn.functional import avg_pool2d, avg_pool3d, interpolate

SUBSAMPLE = {
    2: (lambda x: 4 * avg_pool2d(x, 2)),
    3: (lambda x: 8 * avg_pool3d(x, 2)),
}


def pyramid(img, dim=2, height=32):
    img_s = [img]
    for _ in range(int(math.log2(height))):
        img = SUBSAMPLE[dim](img)
        img_s.append(img)

    img_s.reverse()
    return img_s


def epsilon_schedule(p, diameter, blur, scaling):
    r"""Creates a list of values for the temperature "epsilon" across Sinkhorn iterations.

    We use an aggressive strategy with an exponential cooling
    schedule: starting from a value of :math:`\text{diameter}^p`,
    the temperature epsilon is divided
    by :math:`\text{scaling}^p` at every iteration until reaching
    a minimum value of :math:`\text{blur}^p`.

    Args:
        p (integer or float): The exponent of the Euclidean distance
            :math:`\|x_i-y_j\|` that defines the cost function
            :math:`\text{C}(x_i,y_j) =\tfrac{1}{p} \|x_i-y_j\|^p`.

        diameter (float, positive): Upper bound on the largest distance between
            points :math:`x_i` and :math:`y_j`.

        blur (float, positive): Target value for the entropic regularization
            (":math:`\varepsilon = \text{blur}^p`").

        scaling (float, in (0,1)): Ratio between two successive
            values of the blur scale.

    Returns:
        list of float: list of values for the temperature epsilon.
    """
    eps_list = (
            [diameter ** p]
            + [
                math.exp(e)
                for e in np.arange(
            p * math.log(diameter), p * math.log(blur), p * math.log(scaling)
        )
            ]
            + [blur ** p]
    )
    return eps_list


def convolutional_barycenter_calculation(batch, weights=None, stab_thresh=1e-30, scaling=0.7, need_diffable=False):
    """
    computes the batched convolutional barycenter2d debiased in the log domain
    :param batch: input shape (M, B, C, H, W, [D])
    :param weights: the weight for each sample in the bary center calculation
    :param stab_thresh: default 1e-30, for not dividing by zero.
    :return: the barycenter of the batch with prescribed weights
    """

    def convol_img(_log_img, _kernel):
        _log_img = torch.logsumexp(_kernel[None, None, None, :, :, None] + _log_img[:, :, :, None, ...], dim=-2)
        _log_img = torch.logsumexp(
            _kernel[None, None, None, :, :, None] + _log_img[:, :, :, None, ...].permute(0, 1, 2, 3, -1, -2), dim=-2
        ).permute(0, 1, 2, -1, -2)
        return _log_img

    with torch.no_grad():
        if len(batch.shape) == 6:
            raise NotImplementedError("This method is currently not implemented for 3d")

        nh = batch.shape[0]  # number of histograms for each image
        b = batch.shape[1]  # batch size

        if weights is None:
            weights = 0.5 * torch.ones((nh, 1, 1, 1, 1), dtype=batch.dtype, device=batch.device)

        log_batch = torch.log(batch + stab_thresh)
        log_batch = rearrange(log_batch, 'm b c h w -> (m b) c h w')
        log_img_s = pyramid(log_batch, height=log_batch.shape[-1])[3:]
        log_img_s = [rearrange(log_img, '(m b) c h w -> m b c h w', m=nh, b=b) for log_img in log_img_s]

        _, b, c, h0, w0 = parse_shape(log_img_s[0], 'm b c h w').values()
        db = torch.zeros((b, c, h0, w0), dtype=log_batch.dtype, device=log_batch.device)
        g = torch.zeros(*log_img_s[0].shape, dtype=log_batch.dtype, device=log_batch.device)

        for s, log_img in enumerate(log_img_s):
            n = log_img.shape[-1]
            t = torch.linspace(0, 1, n, dtype=batch.dtype, device=batch.device)
            C = - (t.view(n, 1) - (t.view(1, n))) ** 2
            eps_list = epsilon_schedule(2, 2 / n, 1 / n, scaling)

            for eps in eps_list:
                for _ in range(
                        5 + 15 * (s == len(log_img_s) - 1 and eps == eps_list[-1])
                ):  # do only the last iteration twice
                    m = C / eps
                    log_ku = convol_img(log_img - convol_img(g, m), m)
                    log_bar = db + torch.sum(weights * log_ku, dim=0)

                    db = 0.5 * (db + log_bar - convol_img(db[None], m)[0])

                    g = log_bar[None, ...] - log_ku

            # if not the last scale
            if s != len(log_img_s) - 1:
                # upscale log_bar, c and g
                db = interpolate(db, scale_factor=2, mode='bilinear', align_corners=False)
                g = rearrange(
                    interpolate(
                        rearrange(g, 'm b c h w -> (m b) c h w'), scale_factor=2, mode='bilinear', align_corners=False),
                    '(m b) c h w -> m b c h w', m=nh, b=b
                )
                log_bar = interpolate(log_bar, scale_factor=2, mode='bilinear', align_corners=False)

    if need_diffable:
        with torch.enable_grad():
            log_img = torch.log(batch + stab_thresh)
            log_ku = convol_img(log_img - convol_img(g, m), m)
            log_bar = db + torch.sum(weights * log_ku, dim=0)

    return torch.exp(log_bar)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    square_left = torch.zeros((1, 1, 64, 64), device='cuda')
    square_left[:, :, 24:40, 4:20] = 1
    square_left.requires_grad_(True)

    square_right = torch.zeros((1, 1, 64, 64), device='cuda')
    square_right[:, :, 24:40, 44:60] = 1
    square_right.requires_grad_(True)

    square_middle = torch.zeros((1, 1, 64, 64), device='cuda')
    square_middle[:, :, 24:40, 24:40] = 1
    square_middle.requires_grad_(True)

    _batch = torch.stack([square_left, square_right], dim=0)
    _weights = torch.tensor([0.5, 0.5], device='cuda')
    _weights = _weights[:, None, None, None, None]
    _weights.requires_grad_(True)

    _batch = _batch / torch.sum(_batch, dim=(2,3,4), keepdim=True)

    _barycenter = convolutional_barycenter_calculation(_batch, weights=_weights, need_diffable=True, scaling=0.99)
    print(_barycenter.shape)

    recon_loss = torch.nn.functional.mse_loss(_barycenter, square_middle, reduction='mean')
    [grad_left, grad_right, grad_middle] = torch.autograd.grad(recon_loss, [square_left, square_right, square_middle])

    _, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs[0, 0].imshow(square_left[0, 0].detach().cpu().numpy())
    axs[0, 0].set_title('Left square')

    axs[0, 1].imshow(square_right[0, 0].detach().cpu().numpy())
    axs[0, 1].set_title('Right square')

    axs[0, 2].imshow(square_middle[0, 0].detach().cpu().numpy())
    axs[0, 2].set_title('Middle square')

    axs[0, 3].imshow(_barycenter[0, 0].detach().cpu().numpy())
    axs[0, 3].set_title('Barycenter')

    # plot gradients with the colorbars for each gradient
    im = axs[1, 0].imshow(grad_left[0, 0].detach().cpu().numpy())
    axs[1, 0].set_title('Gradient of left square')
    plt.colorbar(im, ax=axs[1, 0])

    im = axs[1, 1].imshow(grad_right[0, 0].detach().cpu().numpy())
    axs[1, 1].set_title('Gradient of right square')
    plt.colorbar(im, ax=axs[1, 1])

    im = axs[1, 2].imshow(grad_middle[0, 0].detach().cpu().numpy())
    axs[1, 2].set_title('Gradient of middle square')
    plt.colorbar(im, ax=axs[1, 2])

    im = axs[1, 3].imshow(square_middle[0, 0].detach().cpu().numpy() - _barycenter[0, 0].detach().cpu().numpy())
    axs[1, 3].set_title('Difference between middle and bary')
    plt.colorbar(im, ax=axs[1, 3])

    plt.tight_layout()
    plt.show()
