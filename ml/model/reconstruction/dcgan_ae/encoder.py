""""
Used the DCGAN encoder from https://github.com/drorsimon/image_barycenters/blob/master/dcgan_models.py
"""

import torch


class Encoder(torch.nn.Module):
    def __init__(self, latent_dim, num_filters):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        # Hidden layers
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Convolutional layer
            if i == 0:
                conv = torch.nn.Conv2d(1, num_filters[i], kernel_size=4, stride=2, padding=1)
            else:
                conv = torch.nn.Conv2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=1)

            conv_name = 'conv' + str(i + 1)
            self.hidden_layer.add_module(conv_name, conv)

            # Initializer
            torch.nn.init.normal_(conv.weight, mean=0.0, std=0.02)
            torch.nn.init.constant_(conv.bias, 0.0)

            # Batch normalization
            if i > 0:
                bn_name = 'bn' + str(i + 1)
                # self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))
                self.hidden_layer.add_module(bn_name, torch.nn.InstanceNorm2d(num_filters[i]))

            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

        # use adaptive pooling to get the output size
        self.embedding = torch.nn.AdaptiveAvgPool2d((1, 1))

    def return_output_size(self, rand_input):
        return self.forward_pre_linear(rand_input).size()

    def forward_pre_linear(self, rand_input):
        return self.hidden_layer(rand_input)

    def forward(self, x):
        x = self.hidden_layer(x)
        out = self.embedding(x)
        return out


if __name__ == '__main__':
    # Test the encoder
    encoder = Encoder(100, (64, 128, 256, 512))
    print(encoder)
    rand_input = torch.randn((1, 1, 64, 64))
    print(encoder(rand_input).size())
