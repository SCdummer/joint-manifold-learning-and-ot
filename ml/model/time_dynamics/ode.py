import torch
import torch.nn as nn

from einops import rearrange
from torchdiffeq import odeint_adjoint as odeint

from ml.model import utils


class ODEFunc(nn.Module):
    def __init__(self, dim):
        super(ODEFunc, self).__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim),
        )

    def forward(self, t, x):
        return self.net(x)


class ODEBlock(nn.Module):
    def __init__(self, odefunc, timesteps):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.timesteps = timesteps

    def forward(self, x):
        t = torch.linspace(0, 1, self.timesteps).to(x.device)
        x = odeint(self.odefunc, x, t)
        return x


@utils.register_model(dataset='HeLa', name='ode')
class ODE(nn.Module):
    def __init__(self, dim, timesteps):
        super(ODE, self).__init__()
        self.odeblock = ODEBlock(ODEFunc(dim), timesteps)

    def forward(self, x):
        # x will have shape (batch_size, channel, 1, 1) for images
        # rearrange it to (batch_size, channel)
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.odeblock(x)
        x = rearrange(x, 't b c -> t b c () ()')
        return x


if __name__ == '__main__':
    _model = ODE(512, 1)
    _x = torch.randn(64, 512, 1, 1)
    _y = _model(_x)

    print(_y.shape)
