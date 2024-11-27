from .encoder import Encoder
from .decoder import Decoder

import torch

from ml.model import utils


@utils.register_model(dataset='HeLa', name='dcgan_ae')
class HeLaDCGAN(torch.nn.Module):

    def __init__(self, latent_dim=512, num_filters=(64, 128, 256, 512), output_size=(64, 64)):
        super(HeLaDCGAN, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, num_filters)
        self.decoder = Decoder(latent_dim, num_filters, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
