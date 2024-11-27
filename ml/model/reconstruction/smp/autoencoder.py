import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock
from segmentation_models_pytorch.encoders.resnet import resnet_encoders, ResNetEncoder

from ml.model.reconstruction.smp.encoder import ResNet18C, ResNet18T
from ml.model import utils


class MyResNetEncoder(ResNetEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        stages = self.get_stages()

        for i in range(self._depth + 1):
            x = stages[i](x)

        x = F.adaptive_avg_pool2d(x, (1, 1))

        return x


class MyUnetDecoder(torch.nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            last_dimensions,
            output_channels=1,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = [0] * len(decoder_channels)
        out_channels = decoder_channels

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]

        self.blocks = nn.ModuleList(blocks)
        self.last_dimensions = last_dimensions

        self.out = nn.Sequential(
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.ReLU(),
            nn.Conv2d(decoder_channels[-1], output_channels, kernel_size=3, padding=1),
        )

    def forward(self, latent):
        out = F.interpolate(latent, size=self.last_dimensions, mode='bilinear', align_corners=False)
        for i, decoder_block in enumerate(self.blocks):
            out = decoder_block(out)
        return self.out(out)


class AutoEncoder(torch.nn.Module):
    def __init__(
            self,
            encoder_name="resnet18",
            encoder_depth=5,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels: int = 1,
            out_channels: int = 1,
            return_latent=False,
            input_size=(64, 64),
    ):
        super().__init__()

        if encoder_name in resnet_encoders:
            resnet_params = resnet_encoders[encoder_name]['params']
            resnet_params.update(depth=encoder_depth)
            self.encoder = MyResNetEncoder(**resnet_params)
            self.encoder.set_in_channels(in_channels)
        elif encoder_name == "resnet18c":
            self.encoder = ResNet18C(num_channels=in_channels)
        elif encoder_name == "resnet18t":
            self.encoder = ResNet18T(num_channels=in_channels)
        else:
            raise ValueError(f"Unknown encoder name: {encoder_name}")

        self.decoder = MyUnetDecoder(
            encoder_channels=self.encoder._out_channels,
            decoder_channels=decoder_channels,
            output_channels=out_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            attention_type=None,
            last_dimensions=tuple(map(lambda d: d // 32, input_size))
        )

        self.return_latent = return_latent

    def forward(self, x):
        latent = self.encoder(x)
        # upscale the latent to last_dimensions before avg_pooling
        out = self.decoder(latent)
        if self.return_latent:
            return out, latent
        else:
            return out


@utils.register_model(dataset='HeLa', name='ae')
class HeLaAutoEncoder(AutoEncoder):
    def __init__(self):
        super().__init__(
            encoder_name="resnet18t", in_channels=1, return_latent=True, input_size=(64, 64)
        )


if __name__ == '__main__':
    _input_size = (64, 64)
    model = HeLaAutoEncoder()
    x = torch.randn(1, 1, *_input_size)
    y = model(x)
    print(y[0].shape, y[1].shape)
