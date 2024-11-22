""""
Used the DCGAN encoder from https://github.com/drorsimon/image_barycenters/blob/master/dcgan_models.py
"""

import torch


class Encoder(torch.nn.Module):
    def __init__(self, latent_dim, num_filters, img_dims, use_vae=False):
        super(Encoder, self).__init__()

        self.use_vae = use_vae
        self.latent_dim = latent_dim

        # Hidden layers
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Convolutional layer
            if i == 0:
                conv = torch.nn.Conv2d(1, num_filters[i], kernel_size=4, stride=2, padding=1)
            else:
                conv = torch.nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

            conv_name = 'conv' + str(i + 1)
            self.hidden_layer.add_module(conv_name, conv)

            # Initializer
            torch.nn.init.normal_(conv.weight, mean=0.0, std=0.02)
            torch.nn.init.constant_(conv.bias, 0.0)

            # Batch normalization
            if i > 0:
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

        # Save the input dimensions
        self.input_size = img_dims

        # # Output layer
        # self.output_layer = torch.nn.Sequential()
        #
        # # Convolutional layer
        # out = torch.nn.Conv2d(num_filters[i], latent_dim, kernel_size=4, stride=1, padding=0)
        # self.output_layer.add_module('out', out)
        # # Initializer
        # torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        # torch.nn.init.constant_(out.bias, 0.0)
        # # Activation
        # self.output_layer.add_module('bn_out', torch.nn.BatchNorm2d(latent_dim))

        rand_input = torch.randn(img_dims)[None, ...]
        self.output_size = self.return_output_size(rand_input)[1:]
        num_elements = 1
        for i in range(len(self.output_size)):
            num_elements = num_elements * self.output_size[i]

        self.embedding = torch.nn.Conv2d(self.output_size[0], latent_dim, kernel_size=self.output_size[1:])
        self.log_var = torch.nn.Conv2d(self.output_size[0], latent_dim, kernel_size=self.output_size[1:])

    def return_output_size(self, rand_input):
        return self.forward_pre_linear(rand_input).size()

    def forward_pre_linear(self, rand_input):
        return self.hidden_layer(rand_input)

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def reparameterization_trick(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        return z

    def forward(self, x):
        x = self.hidden_layer(x)
        if self.use_vae:
            mu, log_var = self.embedding(x).reshape(-1, self.latent_dim), self.log_var(x).reshape(-1, self.latent_dim)
            out = self.reparameterization_trick(mu, log_var).reshape(-1, self.latent_dim)
        else:
            out = mu = self.embedding(x).reshape(-1, self.latent_dim)
            log_var = None
        return out, mu, log_var