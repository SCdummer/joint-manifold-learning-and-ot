import torch.nn as nn
from torch.utils.hipify.hipify_python import InputError
from torchdiffeq import odeint_adjoint, odeint
import torch
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


nls = {'relu': nn.ReLU,
       'sigmoid': nn.Sigmoid,
       'tanh': nn.Tanh,
       'selu': nn.SELU,
       'softplus': nn.Softplus,
       'gelu': nn.GELU,
       'swish': Swish,
       'elu': nn.ELU}


class VecField(nn.Module): 
    def __init__(self, input_dim, hidden_dims=(64, 64, 64), act='swish'):
        super().__init__()

        if act not in nls.keys():
            raise InputError("The VecField is created with activation function {}, but we only have available: {}".format(act, nls))

        self.act = nls[act]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(self.act())

        for hidden_dim in hidden_dims[1:]:
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(self.act())
            #self.layers.append(nn.BatchNorm1d(hidden_dim))

        self.layers.append(nn.Linear(hidden_dims[-1], input_dim))

    def forward(self, t, z):
        y = z
        for layer in self.layers:
            y = layer(y)
        return y


class NeuralODE(nn.Module):
    def __init__(self, latent_dim, hidden_dims=(64, 64, 64), ode_int_method='euler', act='swish', adjoint_method=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.vel_field = VecField(latent_dim, hidden_dims, act=act)
        self.ode_int_method = ode_int_method
        if self.ode_int_method in ['dopri8', 'dopri5', 'bosh3', 'fehlberg2', 'adaptive_heun']:
            self.adaptive = True
        elif self.ode_int_method in ['euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams']:
            self.adaptive = False
        else:
            raise ValueError("The supplied ode integration method is not supported.")
        if adjoint_method:
            self.odeint = odeint_adjoint
        else:
            self.odeint = odeint

    ### z-latent code, t-marching time, c-place holder for latent code of parameters
    def forward(self, z, t=None, c=None):
        return self.odeint(self.vel_field, z, t, method=self.ode_int_method)

