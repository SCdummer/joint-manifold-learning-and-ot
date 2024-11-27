""""
Taken the Generator (and renamed to Decoder) from https://github.com/drorsimon/image_barycenters/blob/master/dcgan_models.py
"""

import torch


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, num_filters, output_size):
        super(Decoder, self).__init__()

        # Save latent dimensions
        self.latent_dim = latent_dim

        # Hidden layers
        self.hidden_layer = torch.nn.Sequential()
        self.upsample_size = list(map(lambda res: res // 2 ** len(num_filters), output_size))
        self.upsample = torch.nn.ConvTranspose2d(latent_dim, num_filters[0], kernel_size=self.upsample_size)
        for i in range(1, len(num_filters)):
            # Deconvolutional layer
            if i == 0:
                deconv = torch.nn.ConvTranspose2d(latent_dim, num_filters[i], kernel_size=4, stride=1, padding=0)
            else:
                deconv = torch.nn.ConvTranspose2d(
                    num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=1)

            deconv_name = 'deconv' + str(i + 1)
            self.hidden_layer.add_module(deconv_name, deconv)

            torch.nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
            torch.nn.init.constant_(deconv.bias, 0.0)

            # Batch normalization
            bn_name = 'bn' + str(i + 1)
            # self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))
            self.hidden_layer.add_module(bn_name, torch.nn.InstanceNorm2d(num_filters[i]))

            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, torch.nn.ReLU())

        # Output layer
        self.output_layer = torch.nn.Sequential()

        # Deconvolutional layer
        out = torch.nn.ConvTranspose2d(num_filters[i], 1, kernel_size=4, stride=2, padding=1)
        self.output_layer.add_module('out', out)
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out.bias, 0.0)

    def forward(self, x):
        h = self.upsample(x)
        h = self.hidden_layer(h)
        return self.output_layer(h)


if __name__ == '__main__':
    # Test the decoder
    decoder = Decoder(100, (512, 256, 128, 64), (64, 64))
    print(decoder)
    x = torch.randn((1, 100, 1, 1))
    print(decoder(x).shape)
    print('Decoder test successful!')
