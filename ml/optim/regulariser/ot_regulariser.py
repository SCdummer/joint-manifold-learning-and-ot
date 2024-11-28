import math

import torch
import numpy as np

from einops import rearrange, parse_shape
from torch.nn.modules.loss import _Loss
from torch.nn.functional import avg_pool2d, avg_pool3d, interpolate

__all__ = [
    'FastConvolutionalW2Cost'
]

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


def batched_convol_bary_debiased(
        src, tgt, scaling=0.95, stab_thresh=1e-30,
        n_per_eps=1, n_extra_last_scale=4,
        use_pykeops=False, need_diffable=True
):
    """
    computes the batched convolutional barycenter2d debiased in the log domain.
    This code is an adpated version of Python Optimal Transport (POT) library and uses ideas from geomloss library.

    :param src: the source images
    :param tgt: the target images
    :param scaling: the scaling factor for the epsilon schedule
    :param stab_thresh: default 1e-30, for not dividing by zero.
    :param n_per_eps: the number of iterations per epsilon
    :param n_extra_last_scale: the number of extra iterations for the last scale and epsilon
    :param use_pykeops: whether to use pykeops for the computation
    :param need_diffable: whether to return the differentiable version of the cost
    :return: the debiased convolutional wasserstein distance between the source and target images
    """
    if use_pykeops:
        try:
            from pykeops.torch import LazyTensor
        except ImportError:
            raise ImportError("Please install pykeops to use this method, or set use_pykeops=False")

    b = src.shape[0]  # batch size
    c = src.shape[1]  # number of channels

    with torch.no_grad():
        log_src, log_tgt = torch.log(src + stab_thresh), torch.log(tgt + stab_thresh)
        log_src_s = pyramid(log_src, height=min(log_src.shape[-2:]))[-2:]  # consider only the last two scales
        log_tgt_s = pyramid(log_tgt, height=min(log_tgt.shape[-2:]))[-2:]  # consider only the last two scales

        _, _, h0, w0 = parse_shape(log_src_s[0], 'b c h w').values()

        f_aa, g_aa = log_src_s[0], log_src_s[0]
        f_ab, g_ba = log_src_s[0], log_tgt_s[0]
        f_bb, g_bb = log_tgt_s[0], log_tgt_s[0]

        def convol_img(_log_img, _kernel_x, _kernel_y=None):
            if _kernel_y is None:
                _kernel_y = _kernel_x

            if use_pykeops:
                _log_img = LazyTensor(rearrange(_log_img, 'b c h w -> (b c w) () h ()').contiguous())
                _log_img = (_log_img + _kernel_x).logsumexp(dim=2)
                _log_img = LazyTensor(
                    rearrange(_log_img, '(b c w) h () -> (b c h) () w ()', b=b, c=c, w=w_s).contiguous()
                )
                _log_img = (_log_img + _kernel_y).logsumexp(dim=2)
                return rearrange(_log_img, '(b c h) w () -> b c h w', b=b, c=c, h=h_s, w=w_s)
            else:
                _log_img = torch.logsumexp(_kernel_x[None, None, :, :, None] + _log_img[:, :, None, ...], dim=-2)
                _log_img = torch.logsumexp(
                    _kernel_y[None, None, :, :, None] + _log_img[:, :, None, ...].permute(0, 1, 2, -1, -2),
                    dim=-2
                ).permute(0, 1, -1, -2)
                return _log_img

        for s, (s_src, s_tgt) in enumerate(zip(log_src_s, log_tgt_s)):
            h_s, w_s = s_src.shape[-2:]
            t_x = torch.linspace(0, 1, w_s, dtype=src.dtype, device=src.device)
            t_y = torch.linspace(0, 1, h_s, dtype=src.dtype, device=src.device)
            if use_pykeops:
                C_y = - (LazyTensor(t_x.view(1, w_s, 1, 1)) - LazyTensor(t_x.view(1, 1, w_s, 1))) ** 2 / 2
                C_x = - (LazyTensor(t_y.view(1, h_s, 1, 1)) - LazyTensor(t_y.view(1, 1, h_s, 1))) ** 2 / 2
            else:
                C_y = - (t_x.view(w_s, 1) - (t_x.view(1, w_s))) ** 2 / 2
                C_x = - (t_y.view(h_s, 1) - (t_y.view(1, h_s))) ** 2 / 2

            eps_list = epsilon_schedule(2, 2 / max(w_s, h_s), 1 / max(h_s, w_s), scaling)
            # print(h_s, w_s, len(eps_list), eps_list[0], eps_list[-1])

            for eps in eps_list:
                m_x = C_x / eps
                m_y = C_y / eps
                for _ in range(n_per_eps + n_extra_last_scale * (s == len(log_src_s) - 1 and eps == eps_list[-1])):
                    ft_aa = s_src - convol_img(g_aa, m_x, m_y)
                    gt_aa = s_src - convol_img(f_aa, m_x, m_y)
                    f_aa, g_aa = (f_aa + ft_aa) / 2, (g_aa + gt_aa) / 2

                    ft_bb = s_tgt - convol_img(g_bb, m_x, m_y)
                    gt_bb = s_tgt - convol_img(f_bb, m_x, m_y)
                    f_bb, g_bb = (f_bb + ft_bb) / 2, (g_bb + gt_bb) / 2

                    ft_ab = s_src - convol_img(g_ba, m_x, m_y)
                    gt_ba = s_tgt - convol_img(f_ab, m_x, m_y)
                    f_ab, g_ba = (f_ab + ft_ab) / 2, (g_ba + gt_ba) / 2

            # if not the last scale
            if s != len(log_src_s) - 1:
                # upscale
                variables_to_upscale = [f_aa, f_ab, f_bb, g_aa, g_ba, g_bb]
                for i in range(len(variables_to_upscale)):
                    variables_to_upscale[i] = interpolate(
                        variables_to_upscale[i], scale_factor=2, mode='bilinear', align_corners=False
                    )
                f_aa, f_ab, f_bb, g_aa, g_ba, g_bb = variables_to_upscale

    if need_diffable:
        with torch.enable_grad():
            log_src = torch.log(src + stab_thresh)
            log_tgt = torch.log(tgt + stab_thresh)

            ft_aa = log_src - convol_img(g_aa, m_x, m_y)
            f_aa = (f_aa + ft_aa) / 2

            gt_bb = log_tgt - convol_img(f_bb, m_x, m_y)
            g_bb = (g_bb + gt_bb) / 2

            ft_ab = log_src - convol_img(g_ba, m_x, m_y)
            gt_ba = log_tgt - convol_img(f_ab, m_x, m_y)
            f_ab, g_ba = (f_ab + ft_ab) / 2, (g_ba + gt_ba) / 2

    s_dist = src / src.sum(dim=(-1, -2, -3), keepdim=True)
    t_dist = tgt / tgt.sum(dim=(-1, -2, -3), keepdim=True)

    return torch.reshape((f_ab - f_aa) * s_dist + (g_ba - g_bb) * t_dist, (b, c, -1)).mean(-1)


class FastConvolutionalW2Cost(_Loss):

    def __init__(
            self, reduction='mean', scaling=0.95, use_pykeops=False, n_per_eps=1, n_extra_last_scale=4,
    ):
        super(FastConvolutionalW2Cost, self).__init__(reduction=reduction)
        self.scaling = scaling
        self.use_pykeops = use_pykeops
        self.n_per_eps = n_per_eps
        self.n_extra_last_scale = n_extra_last_scale

    def forward(self, source, target):

        cost = batched_convol_bary_debiased(
            source, target, scaling=self.scaling,
            n_per_eps=self.n_per_eps, n_extra_last_scale=self.n_extra_last_scale,
            use_pykeops=self.use_pykeops, need_diffable=True
        )
        if self.reduction == 'mean':
            return cost.mean()
        elif self.reduction == 'sum':
            return cost.sum()
        else:
            return cost
