""""
Used the DCGAN encoder from https://github.com/drorsimon/image_barycenters/blob/master/dcgan_models.py
"""

import torch

class Encoder(torch.nn.Module):
    def __init__(self, latent_dim, num_filters, img_dims):
        super(Encoder, self).__init__()

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

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Convolutional layer
        out = torch.nn.Conv2d(num_filters[i], latent_dim, kernel_size=4, stride=1, padding=0)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('bn_out', torch.nn.BatchNorm2d(latent_dim))
        self.input_size = img_dims
        rand_input = torch.randn(img_dims)[None, ...]
        self.output_size = self.return_output_size(rand_input)[-2:]

    def return_output_size(self, rand_input):
        return self.forward(rand_input).size()

    def forward(self, x):
        x = self.hidden_layer(x)
        out = self.output_layer(x)
        return out