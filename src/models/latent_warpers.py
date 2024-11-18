import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class VecField(nn.Module): 
    def __init__(self, input_dim, hidden_dims=(64, 64, 64)):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())

        for hidden_dim in hidden_dims[1:]:
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_dim))

        self.layers.append(nn.Linear(hidden_dims[-1], input_dim))

    def forward(self, t, z):
        y = z
        for layer in self.layers:
            y = layer(y)
        return y


class NeuralODE(nn.Module):
    def __init__(self, latent_dim, hidden_dims=(64, 64, 64), ode_int_method='euler', step_size=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.vel_field = VecField(latent_dim, hidden_dims)
        self.ode_int_method = ode_int_method
        self.step_size = step_size
        if self.ode_int_method in ['dopri8', 'dopri5', 'bosh3', 'fehlberg2', 'adaptive_heun']:
            self.adaptive = True
        elif self.ode_int_method in ['euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams']:
            self.adaptive = False
            if step_size is None:
                raise ValueError('time_steps has to be provided')
        else:
            raise ValueError("The supplied ode integration method is not supported.")

    ### z-latent code, t-marching time, c-place holder for latent code of parameters
    def forward(self, z, t=None, c=None):
        return odeint(self.vel_field, z, t, method=self.ode_int_method)

