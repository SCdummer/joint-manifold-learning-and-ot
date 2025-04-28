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


def convolutional_barycenter_calculation(batch, weights=None, stab_thresh=1e-30, scaling=0.7):
    """
    computes the batched convolutional barycenter2d debiased in the log domain
    :param batch: input shape (M, B, C, H, W, [D])
    :param weights: the weight for each sample in the bary center calculation
    :param stab_thresh: default 1e-30, for not dividing by zero.
    :return: the barycenter of the batch with prescribed weights
    """

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

    def convol_img(_log_img, _kernel):
        _log_img = torch.logsumexp(_kernel[None, None, None, :, :, None] + _log_img[:, :, :, None, ...], dim=-2)
        _log_img = torch.logsumexp(
            _kernel[None, None, None, :, :, None] + _log_img[:, :, :, None, ...].permute(0, 1, 2, 3, -1, -2), dim=-2
        ).permute(0, 1, 2, -1, -2)
        return _log_img

    for s, log_img in enumerate(log_img_s):
        n = log_img.shape[-1]
        t = torch.linspace(0, 1, n, dtype=batch.dtype, device=batch.device)
        C = - (t.view(n, 1) - (t.view(1, n))) ** 2
        eps_list = epsilon_schedule(2, 2 / n, 1 / n, scaling)

        for eps in eps_list:
            for _ in range(1 + (s == len(log_img_s) - 1 and eps == eps_list[-1])):  # do only the last iteration twice
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
                interpolate(rearrange(g, 'm b c h w -> (m b) c h w'), scale_factor=2, mode='bilinear',
                            align_corners=False),
                '(m b) c h w -> m b c h w', m=nh, b=b
            )
            log_bar = interpolate(log_bar, scale_factor=2, mode='bilinear', align_corners=False)

    return torch.exp(log_bar)
