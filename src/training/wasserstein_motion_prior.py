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
        batch, scaling=0.9, weights=None, stab_thresh=1e-30, blur=None, use_pykeops=False, return_cost=False, need_diffable=False
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
        log_img_s = pyramid(log_batch, height=min(log_batch.shape[-2:]))[3:]
        log_img_s = [rearrange(log_img, '(m b) c h w -> m b c h w', m=nh, b=b) for log_img in log_img_s]

        _, b, c, h0, w0 = parse_shape(log_img_s[0], 'm b c h w').values()
        db = torch.zeros((b, c, h0, w0), dtype=log_batch.dtype, device=log_batch.device)
        g = torch.zeros(*log_img_s[0].shape, dtype=log_batch.dtype, device=log_batch.device,
                        requires_grad=batch.requires_grad)

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
                _log_img = torch.logsumexp(_kernel_x[None, None, None, :, :, None] + _log_img[:, :, :, None, ...],
                                           dim=-2)
                _log_img = torch.logsumexp(
                    _kernel_y[None, None, None, :, :, None] + _log_img[:, :, :, None, ...].permute(0, 1, 2, 3, -1, -2),
                    dim=-2
                ).permute(0, 1, 2, -1, -2)
                return _log_img

        for s, log_img in enumerate(log_img_s):
            h_s, w_s = log_img.shape[-2:]
            t_x = torch.linspace(0, 1, w_s, dtype=batch.dtype, device=batch.device) #torch.arange(0, w_s, dtype=batch.dtype, device=batch.device) #torch.linspace(0, 1, w_s, dtype=batch.dtype, device=batch.device)
            t_y = torch.linspace(0, 1, h_s, dtype=batch.dtype, device=batch.device) #torch.arange(0, h_s, dtype=batch.dtype, device=batch.device) #torch.linspace(0, 1, h_s, dtype=batch.dtype, device=batch.device)
            if use_pykeops:
                C_y = - (LazyTensor(t_x.view(1, w_s, 1, 1)) - LazyTensor(t_x.view(1, 1, w_s, 1))) ** 2
                C_x = - (LazyTensor(t_y.view(1, h_s, 1, 1)) - LazyTensor(t_y.view(1, 1, h_s, 1))) ** 2
            else:
                C_y = - (t_x.view(w_s, 1) - (t_x.view(1, w_s))) ** 2
                C_x = - (t_y.view(h_s, 1) - (t_y.view(1, h_s))) ** 2

            if blur is None:
                blur = 1 / max(h_s, w_s)
            eps_list = epsilon_schedule(2, 2 / max(w_s, h_s), 1 / max(h_s, w_s), scaling)
            # print(n, len(eps_list), eps_list[0], eps_list[-1])

            for eps in eps_list:
                m_x = C_x / eps
                m_y = C_y / eps
                for _ in range(1 + (s == len(log_img_s) - 1)):  # do the last iteration twice
                    f = log_img - convol_img(g, m_x, m_y)
                    log_ku = convol_img(f, m_x, m_y)
                    log_bar = db + torch.sum(weights * log_ku, dim=0)

                    db = 0.5 * (db + log_bar - convol_img(db[None], m_x, m_y)[0])

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

    if need_diffable:
        with torch.enable_grad():
            f = torch.log(batch + stab_thresh) - convol_img(g, m_x, m_y)
            log_ku = convol_img(f, m_x, m_y)
            log_bar = db + torch.sum(weights * log_ku, dim=0)
            g = log_bar[None, ...] - log_ku

    if return_cost:
        dist = batch / batch.sum(dim=(2, 3, 4), keepdim=True)
        return torch.exp(log_bar), torch.reshape((f[0] * dist[0] + g[0] * dist[1]), (b, -1)).mean(dim=1)
    else:
        return torch.exp(log_bar)


class FastConvolutionalW2Cost(_Loss):
    def __init__(self, reduction='mean', scaling=0.9, blur=None, use_pykeops=False):
        super(FastConvolutionalW2Cost, self).__init__(reduction=reduction)
        self.scaling = scaling
        self.use_pykeops = use_pykeops
        self.blur = blur

    def forward(self, source, target):
        stacked = torch.stack([source, target], dim=0)

        _, cost = batched_convol_bary_debiased(
            stacked, scaling=self.scaling, blur=self.blur, return_cost=True, need_diffable=True, use_pykeops=self.use_pykeops
        )
        if self.reduction == 'mean':
            return cost.mean()
        elif self.reduction == 'sum':
            return cost.sum()
        else:
            return cost


def create_circle(_translation, inner_rad=3, val=0.5):
    _image_of_circle = torch.zeros(32, 64)
    _y, _x = torch.meshgrid(
        torch.arange(32, dtype=torch.float32),
        torch.arange(64, dtype=torch.float32),
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
    _image_of_square = torch.zeros(32, 64)
    # _image_of_square = torch.rand(32, 64)
    _image_of_square[12:20, 44:52] = 1
    _image_of_square = _image_of_square.unsqueeze(0)

    # generate an image of a circle
    _image_of_circle = create_circle(0.0)

    # show the images
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(_image_of_square.cpu().numpy().transpose(1, 2, 0), cmap='gray')
    axes[1].imshow(_image_of_circle.cpu().numpy().transpose(1, 2, 0), cmap='gray')
    plt.tight_layout()
    plt.show()

    # stack the images
    _source = torch.cat([_image_of_square.unsqueeze(0), _image_of_circle.unsqueeze(0)], dim=0).cuda()
    _source.requires_grad_(True)

    _loss = FastConvolutionalW2Cost(scaling=0.9, reduction='none')

    _target = _source.roll(1, 0)
    _cost = _loss(_source, _target)

    [gi_0] = torch.autograd.grad(_cost[0], _source, retain_graph=True)
    [gi_1] = torch.autograd.grad(_cost[1], _source, retain_graph=False)

    _, ax = plt.subplots(1, 2, figsize=(15, 5))
    # compute the direction of descent at each point in the gi
    ax[0].set_title(f'{_cost[0].item():.2f}')
    ax[0].imshow(gi_0[0].detach().cpu().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].set_title(f'{_cost[1].item():.2f}')
    ax[1].imshow(gi_1[1].detach().cpu().numpy().transpose(1, 2, 0))
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

    # stack the images
    _source = torch.cat([_image_of_square.unsqueeze(0), _image_of_circle.unsqueeze(0)], dim=0).cuda()
    _source.requires_grad_(True)

    _loss = FastConvolutionalW2Cost(scaling=0.9, reduction='mean')

    _target = _source.roll(1, 0)
    _cost = _loss(_source, _target)

    [gi] = torch.autograd.grad(_cost, _source, retain_graph=False)

    _, ax = plt.subplots(1, 2, figsize=(15, 5))
    # compute the direction of descent at each point in the gi
    ax[0].set_title(f'{_cost.item():.2f}')
    ax[0].imshow(gi_0[0].detach().cpu().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].set_title(f'{_cost.item():.2f}')
    ax[1].imshow(gi_1[1].detach().cpu().numpy().transpose(1, 2, 0))
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

    # Testing if the code works for circles that are very close
    circle_1 = create_circle(0.0, inner_rad=1.5, val=0.5)
    circle_2 = create_circle(0.0, inner_rad=3, val=1)
    _, axes = plt.subplots(1, 2)
    circle_1 = circle_1 / circle_1.sum()
    circle_2 = circle_2 / circle_2.sum()
    axes[0].imshow(circle_1.cpu().numpy().transpose(1, 2, 0), cmap='gray')
    axes[1].imshow(circle_2.cpu().numpy().transpose(1, 2, 0), cmap='gray')
    plt.tight_layout()
    plt.show()

    # Test the calculation by comparison to python optmal transport (POT)
    import ot
    import geomloss
    unit_square_grid = False

    # Get the pairwise euclidean distances
    if unit_square_grid:
        _y, _x = np.meshgrid(
            np.arange(32, dtype=np.float32) / 31.0,
            np.arange(64, dtype=np.float32) / 64.0,
            indexing='ij'
        )
    else:
        _y, _x = np.meshgrid(
            np.arange(32, dtype=np.float32),
            np.arange(64, dtype=np.float32),
            indexing='ij'
        )

    # Concatenate the things
    yx = np.concatenate([_y[..., None], _x[..., None]], axis=-1)

    # Get the flattened images and grid
    yx_flat = np.reshape(yx, (-1, 2))
    img1_flat = np.reshape(circle_1.cpu().numpy().squeeze(), (-1,))
    img2_flat = np.reshape(circle_2.cpu().numpy().squeeze(), (-1,))

    # Normalize the images
    img1_flat = img1_flat / img1_flat.sum()
    img2_flat = img2_flat / img2_flat.sum()

    # Calculate the cost matrix
    C = ot.dist(yx_flat, yx_flat, metric='euclidean')

    # Also calculate some things necessary for geomloss
    w_s, h_s = 32, 64
    sinkhorn_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=1 / max(h_s, w_s), diameter=2 / max(w_s, h_s), scaling=0.9, debias=True)
    geomloss_input_1 = torch.tensor(img1_flat)
    geomloss_input_2 = torch.tensor(img2_flat)
    grid = torch.tensor(yx_flat)
    geomloss_val = sinkhorn_loss(geomloss_input_1, grid, geomloss_input_2, grid)

    # Calculate the OT cost
    loss = ot.emd2(img1_flat, img2_flat, C)
    loss_sinkhorn = ot.sinkhorn2(img1_flat, img2_flat, C, 1 / max(h_s, w_s))
    print("Python Optimal Transport: ", loss)
    print("Python Optimal Transport sinkhorn:", loss_sinkhorn)
    print("GeomLoss: ", geomloss_val)
    print("Own debiased convolutional wasserstein distance:", _loss(circle_1[None, ...], circle_2[None, ...]))
    print("Finished running the tests...")