""""
Taken the Generator (and renamed to Decoder) from https://github.com/drorsimon/image_barycenters/blob/master/dcgan_models.py
"""

import torch

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, num_filters, upsample_size, most_common_val=0.0):
        super(Decoder, self).__init__()

        # Save latent dimensions
        self.latent_dim = latent_dim

        # Hidden layers
        self.hidden_layer = torch.nn.Sequential()
        self.upsample_size = upsample_size
        self.upsample = torch.nn.ConvTranspose2d(latent_dim, num_filters[0], kernel_size=self.upsample_size[1:])
        for i in range(1, len(num_filters)):
            # Deconvolutional layer
            if i == 0:
                deconv = torch.nn.ConvTranspose2d(latent_dim, num_filters[i], kernel_size=4, stride=1, padding=0)
            else:
                deconv = torch.nn.ConvTranspose2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2,
                                                  padding=1)

            deconv_name = 'deconv' + str(i + 1)
            self.hidden_layer.add_module(deconv_name, deconv)

            torch.nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
            torch.nn.init.constant_(deconv.bias, 0.0)

            # Batch normalization
            bn_name = 'bn' + str(i + 1)
            #self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))
            self.hidden_layer.add_module(bn_name, torch.nn.InstanceNorm2d(num_filters[i]))

            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, torch.nn.ReLU())

        # Output layer
        self.output_layer = torch.nn.Sequential()

        # Deconvolutional layer
        self.most_common_val = most_common_val
        if self.most_common_val == 0.0:
            out = torch.nn.ConvTranspose2d(num_filters[i], 1, kernel_size=4, stride=2, padding=1)
        else:
            out = torch.nn.ConvTranspose2d(num_filters[i], 2, kernel_size=4, stride=2, padding=1)
        self.output_layer.add_module('out', out)
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out.bias, 0.0)

        # Activation
        self.output_layer.add_module('act', torch.nn.Sigmoid())

    def upsample_vector_to_matrix(self, v):
        #return v.reshape((-1, *self.upsample_size))
        return v.reshape((-1, self.latent_dim, 1, 1))

    def forward(self, x):
        x = self.upsample(self.upsample_vector_to_matrix(x))
        x = self.hidden_layer(x)
        out = self.output_layer(x)
        return out
        # if self.most_common_val == 0.0:
        #     return out
        # else:
        #     out = self.most_common_val + (1.0 - self.most_common_val) * out[:, 0, ...] - self.most_common_val * out[:, 1, ...]
        #     return out[:, None, ...]
