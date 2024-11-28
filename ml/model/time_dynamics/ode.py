import torch
import torch.nn as nn

from einops import rearrange
from torchdiffeq import odeint

from ml.model import utils


class ODEFunc(nn.Module):
    def __init__(self, dim):
        super(ODEFunc, self).__init__()
        self.dim = dim

        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, dim),
        )

    def forward(self, t, x):
        return self.net(x)


class ODEBlock(nn.Module):
    def __init__(self, odefunc, timesteps, step=0.01):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.timesteps = timesteps
        self.h = step

    def forward(self, x):
        t_actual = torch.linspace(0, 1, self.timesteps, device=x.device)
        t_rand = torch.rand((1,), device=x.device) * (1 - self.h - 1e-6) + (self.h / 2 + 1e-6)
        t_rand = torch.tensor([t_rand - self.h / 2, t_rand + self.h / 2], device=x.device)
        # insert the random time step such that is monotonically increasing
        t = torch.cat([t_actual, t_rand])
        t, indices = torch.sort(t)
        x = odeint(self.odefunc, x, t, method='dopri5')
        # get the results at the random time step
        x_t_rand_step = x[indices >= len(t_actual)]
        x = x[indices < len(t_actual)]
        return x, x_t_rand_step


@utils.register_model(dataset='HeLa', name='ode')
class ODE(nn.Module):
    def __init__(self, dim, timesteps):
        super(ODE, self).__init__()
        self.odeblock = ODEBlock(ODEFunc(dim), timesteps)

    def forward(self, x):
        # x will have shape (batch_size, channel, 1, 1) for images
        # rearrange it to (batch_size, channel)
        x = x.squeeze(-1).squeeze(-1)
        xt, xt_rand_step = self.odeblock(x)
        xt = xt.unsqueeze(-1).unsqueeze(-1)
        xt_rand_step = xt_rand_step.unsqueeze(-1).unsqueeze(-1)
        return xt, xt_rand_step


if __name__ == '__main__':
    _model = ODE(512, 10)
    _x = torch.randn(64, 512, 1, 1)
    _y = _model(_x)

    print(_y[0].shape, _y[1].shape)
