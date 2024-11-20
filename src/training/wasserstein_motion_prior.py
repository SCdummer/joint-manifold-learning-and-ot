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
        batch, scaling=0.95, weights=None, stab_thresh=1e-30, use_pykeops=False, return_cost=False, need_diffable=False,
        n_per_eps=1, n_extra_last_scale=4
):
    """
    computes the batched convolutional barycenter2d debiased in the log domain.
    This code is an adpated version of Python Optimal Transport (POT) library and uses ideas from geomloss library.

    :param batch: input shape (2, B, C, H, W, [D])
    :param weights: the weight for each sample in the bary center calculation
    :param stab_thresh: default 1e-30, for not dividing by zero.
    :return: the batched convolutional barycenter2d debiased images with specified weights
    """
    if use_pykeops:
        try:
            from pykeops.torch import LazyTensor
        except ImportError:
            raise ImportError("Please install pykeops to use this method, or set use_pykeops=False")

    if len(batch.shape) == 6:
        raise NotImplementedError("This method is currently not implemented for 3d")

    nh = batch.shape[0]  # number of histograms for each image

    assert nh == 2 and return_cost, \
        "This method is currently implemented for 2 histograms only, one source and one target when return_cost=True"

    b = batch.shape[1]  # batch size

    if weights is None:
        weights = torch.zeros((nh, 1, 1, 1, 1), dtype=batch.dtype, device=batch.device)
        weights[1] = 1  # the weight for the target histogram is one

    # check if weights are binary
    if return_cost and not torch.all(weights[torch.nonzero(weights, as_tuple=True)].eq(1.)):
        raise ValueError("Weights must be binary for return_cost=True")

    with torch.no_grad():
        log_batch = torch.log(batch + stab_thresh)
        log_batch = rearrange(log_batch, 'm b c h w -> (m b) c h w')
        log_img_s = pyramid(log_batch, height=min(log_batch.shape[-2:]))[-2:]  # consider only the last two scales
        log_img_s = [rearrange(log_img, '(m b) c h w -> m b c h w', m=nh, b=b) for log_img in log_img_s]

        _, b, c, h0, w0 = parse_shape(log_img_s[0], 'm b c h w').values()

        db_aa = torch.zeros((b, c, h0, w0), dtype=log_batch.dtype, device=log_batch.device)
        db = torch.zeros((b, c, h0, w0), dtype=log_batch.dtype, device=log_batch.device)
        db_bb = torch.zeros((b, c, h0, w0), dtype=log_batch.dtype, device=log_batch.device)

        g_aa = log_img_s[0][0]
        g_ba = log_img_s[0]
        g_bb = log_img_s[0][1]

        def convol_img(_log_img, _kernel_x, _kernel_y=None):
            if _kernel_y is None:
                _kernel_y = _kernel_x

            if use_pykeops:
                # number of histograms for each image, in some cases this might be one since we update debiasing term db
                _nh = _log_img.shape[0]

                _log_img = LazyTensor(rearrange(_log_img, 'm b c h w -> (m b c w) () h ()'))
                _log_img = (_log_img + _kernel_x).logsumexp(dim=2)
                _log_img = LazyTensor(
                    rearrange(_log_img, '(m b c w) h () -> (m b c h) () w ()', m=_nh, b=b, c=c, w=w_s))
                _log_img = (_log_img + _kernel_y).logsumexp(dim=2)
                return rearrange(_log_img, '(m b c h) w () -> m b c h w', m=_nh, b=b, c=c, h=h_s, w=w_s)
            else:
                _log_img = torch.logsumexp(_kernel_x[None, None, None, :, :, None] + _log_img[:, :, :, None, ...], dim=-2)
                _log_img = torch.logsumexp(
                    _kernel_y[None, None, None, :, :, None] + _log_img[:, :, :, None, ...].permute(0, 1, 2, 3, -1, -2),
                    dim=-2
                ).permute(0, 1, 2, -1, -2)
                return _log_img

        for s, log_img in enumerate(log_img_s):
            h_s, w_s = log_img.shape[-2:]
            t_x = torch.linspace(0, 1, w_s, dtype=batch.dtype, device=batch.device)
            t_y = torch.linspace(0, 1, h_s, dtype=batch.dtype, device=batch.device)
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
                for _ in range(n_per_eps + n_extra_last_scale * (s == len(log_img_s) - 1 and eps == eps_list[-1])):  # do only the last iteration 5 times
                    f_aa = log_img[0] - convol_img(g_aa[None], m_x, m_y)[0]
                    log_ku = convol_img(f_aa[None], m_x, m_y)[0]
                    log_bar_aa = db_aa + log_ku
                    db_aa = 0.5 * (db_aa + log_bar_aa - convol_img(db_aa[None], m_x, m_y)[0])
                    g_aa = log_bar_aa - log_ku

                    f_bb = log_img[1] - convol_img(g_bb[None], m_x, m_y)[0]
                    log_ku = convol_img(f_bb[None], m_x, m_y)[0]
                    log_bar_bb = db_bb + log_ku
                    db_bb = 0.5 * (db_bb + log_bar_bb - convol_img(db_bb[None], m_x, m_y)[0])
                    g_bb = log_bar_bb - log_ku

                    f_ab = log_img - convol_img(g_ba, m_x, m_y)
                    log_ku = convol_img(f_ab, m_x, m_y)
                    log_bar = db + torch.sum(weights * log_ku, dim=0)
                    db = 0.5 * (db + log_bar - convol_img(db[None], m_x, m_y)[0])
                    g_ba = log_bar[None, ...] - log_ku

            # if not the last scale
            if s != len(log_img_s) - 1:
                # upscale log_bar, c and g
                db_aa = interpolate(db_aa, scale_factor=2, mode='bilinear', align_corners=False)
                db = interpolate(db, scale_factor=2, mode='bilinear', align_corners=False)
                db_bb = interpolate(db_bb, scale_factor=2, mode='bilinear', align_corners=False)
                g_ba = rearrange(
                    interpolate(
                        rearrange(g_ba, 'm b c h w -> (m b) c h w'),
                        scale_factor=2, mode='bilinear', align_corners=False
                    ),
                    '(m b) c h w -> m b c h w', m=nh, b=b
                )
                g_aa = interpolate(g_aa, scale_factor=2, mode='bilinear', align_corners=False)
                g_bb = interpolate(g_bb, scale_factor=2, mode='bilinear', align_corners=False)
                log_bar = interpolate(log_bar, scale_factor=2, mode='bilinear', align_corners=False)

    if need_diffable:
        with torch.enable_grad():
            log_img = torch.log(batch + stab_thresh)

            f_aa = log_img[0] - convol_img(g_aa[None], m_x, m_y)[0]
            log_ku = convol_img(f_aa[None], m_x, m_y)[0]
            log_bar_aa = db_aa + log_ku
            g_aa = log_bar_aa - log_ku
            f_aa = log_img[0] - convol_img(g_aa[None], m_x, m_y)[0]

            f_bb = log_img[1] - convol_img(g_bb[None], m_x, m_y)[0]
            log_ku = convol_img(f_bb[None], m_x, m_y)[0]
            log_bar_bb = db_bb + log_ku
            g_bb = log_bar_bb - log_ku

            f_ab = log_img - convol_img(g_ba, m_x, m_y)
            log_ku = convol_img(f_ab, m_x, m_y)
            log_bar = db + torch.sum(weights * log_ku, dim=0)
            g_ba = log_bar[None, ...] - log_ku

    if return_cost:
        return torch.exp(log_bar), torch.reshape((f_ab[0] - f_aa) * batch[0] + (g_ba[0] - g_bb) * batch[1], (b, -1)).mean(-1)
    else:
        return torch.exp(log_bar)


class FastConvolutionalW2Cost(_Loss):
    def __init__(self, reduction='mean', scaling=0.95, use_pykeops=False, n_per_eps=1, n_extra_last_scale=4):
        super(FastConvolutionalW2Cost, self).__init__(reduction=reduction)
        self.scaling = scaling
        self.use_pykeops = use_pykeops
        self.n_per_eps = n_per_eps
        self.n_extra_last_scale = n_extra_last_scale

    def forward(self, source, target):
        stacked = torch.stack([source, target], dim=0)

        image, cost = batched_convol_bary_debiased(
            stacked, scaling=self.scaling, return_cost=True, need_diffable=True, use_pykeops=self.use_pykeops,
            n_per_eps=self.n_per_eps, n_extra_last_scale=self.n_extra_last_scale
        )
        if self.reduction == 'mean':
            return image, cost.mean()
        elif self.reduction == 'sum':
            return image, cost.sum()
        else:
            return image, cost


def create_circle(_translation, inner_rad=3., val=0.5):
    _image_of_circle = torch.zeros(32, 32)
    _y, _x = torch.meshgrid(
        torch.arange(32, dtype=torch.float32),
        torch.arange(32, dtype=torch.float32),
        indexing='ij'
    )
    _circle_inner = (_x - 24 - _translation) ** 2 + (_y - 24 - _translation) ** 2 <= inner_rad ** 2
    _circle_outer = ((_x - 24 - _translation) ** 2 + (_y - 24 - _translation) ** 2 <= 3 ** 2) & ((_x - 24 - _translation) ** 2 + (_y - 24 - _translation) ** 2 > inner_rad ** 2)
    _circle_out = _circle_outer
    _image_of_circle[_circle_inner] = 1
    _image_of_circle[_circle_outer] = val
    _circle_inner = (_x - 8 - _translation) ** 2 + (_y - 8 - _translation) ** 2 <= inner_rad ** 2
    _circle_outer = ((_x - 8 - _translation) ** 2 + (_y - 8 - _translation) ** 2 <= 3 ** 2) & ((_x - 8 - _translation) ** 2 + (_y - 8 - _translation) ** 2 > inner_rad ** 2)
    _image_of_circle[_circle_inner] = 1
    _image_of_circle[_circle_outer] = val
    _image_of_circle = _image_of_circle.unsqueeze(0)
    return _image_of_circle



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # generate an image of a square
    # _image_of_square = torch.zeros(32, 32)
    # _image_of_square[0, 0] = 1
    # _image_of_square = torch.rand(32, 32)
    # _image_of_square[8:24, 8:24] = 1
    # _image_of_square = _image_of_square.unsqueeze(0)
    _image_of_square = create_circle(0, inner_rad=3.0, val=0.5)
    # _image_of_square = _image_of_square / torch.sum(_image_of_square, dim=(-1, -2), keepdim=True)

    # # generate an image of a circle
    # _image_of_circle = torch.zeros(32, 32)
    # _image_of_circle[-1, -1] = 1
    # _y, _x = torch.meshgrid(
    #     torch.arange(32, dtype=torch.float32),
    #     torch.arange(32, dtype=torch.float32),
    #     indexing='ij'
    # )
    # _circle = (_x - 24) ** 2 + (_y - 24) ** 2 <= 3 ** 2
    # _image_of_circle[_circle] = 1
    # _circle = (_x - 8) ** 2 + (_y - 8) ** 2 <= 3 ** 2
    # _image_of_circle[_circle] = 1

    # _image_of_circle = _image_of_circle.unsqueeze(0)

    _image_of_circle = create_circle(2, inner_rad=3.0, val=0.5)
    # _image_of_circle = _image_of_circle / torch.sum(_image_of_circle, dim=(-1, -2), keepdim=True)

    # # show the images
    # _, axes = plt.subplots(1, 2)
    # axes[0].imshow(_image_of_square.cpu().numpy().transpose(1, 2, 0), cmap='gray')
    # axes[1].imshow(_image_of_circle.cpu().numpy().transpose(1, 2, 0), cmap='gray')
    # plt.tight_layout()
    # plt.show()

    # # stack the images
    # # _source = torch.rand(2, 1, 32, 32, dtype=torch.float32, device='cuda')
    # _source = torch.cat([_image_of_square.unsqueeze(0), _image_of_circle.unsqueeze(0)], dim=0).cuda()
    # _source.requires_grad_(True)
    # _target = torch.cat([_image_of_circle.unsqueeze(0), create_circle(0, inner_rad=1.5, val=0.5).unsqueeze(0)], dim=0).cuda()

    _source = _image_of_square.unsqueeze(0).cuda()
    # _source = torch.rand(1, 1, 32, 32, dtype=torch.float32, device='cuda')
    _source.requires_grad_(True)
    _target = _image_of_circle.unsqueeze(0).cuda()

    _loss = FastConvolutionalW2Cost(scaling=0.9, reduction='none')
    _image_ab, _cost = _loss(_source, _target)

    # _, axs = plt.subplots(1, 3, figsize=(15, 5))
    # axs[0].imshow(_image_ab.squeeze().detach().cpu().numpy())
    # axs[0].set_title("Image AB")
    # axs[1].imshow(_image_aa.squeeze().detach().cpu().numpy())
    # axs[1].set_title("Image AA")
    # axs[2].imshow(_image_bb.squeeze().detach().cpu().numpy())
    # axs[2].set_title("Image BB")
    # plt.tight_layout()
    # plt.show()
    #
    # _cost = torch.reshape((f_ab[0] - f_aa[0]) * _source + (g_ba[0] - g_bb[0]) * _target, (1, -1)).mean(-1)

    [_g] = torch.autograd.grad(_cost.mean(), [_source], retain_graph=False)

    print(_g.shape)

    # plot the gradient and the image
    _, axs = plt.subplots(1, 4, figsize=(15, 5))
    plt.colorbar(axs[0].imshow(_g.squeeze().detach().cpu().numpy()), ax=axs[0])
    axs[0].set_title("Gradient")
    axs[1].imshow(_source.squeeze().detach().cpu().numpy())
    axs[1].set_title("Source")
    axs[2].imshow(_target.squeeze().cpu().numpy())
    axs[2].set_title("Target")
    axs[3].imshow(_image_ab.squeeze().detach().cpu().numpy())
    axs[3].set_title("Computed Image")
    plt.suptitle(f"Cost: {_cost.mean().item():.4f}")
    plt.tight_layout()
    plt.show()
