import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union
import numpy as np


class NLayerDiscriminator(nn.Sequential):
    def __init__(self,
                 in_channels,
                 n_layers=3,
                 multiplier=64,
                 kernel_size=4,
                 padding=1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, ):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers = [
            nn.Conv2d(in_channels, multiplier, kernel_size=kernel_size, stride=2, padding=padding, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ]

        channels = 2 ** np.arange(n_layers + 1)
        channels = np.minimum(channels, 8)
        channels = multiplier * channels

        downsampling_channels = channels[:-1]
        for in_channels, out_channels in zip(downsampling_channels[:-1], downsampling_channels[1:]):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
                norm_layer(out_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]

        in_channels, out_channels = channels[-2:]
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ]

        layers += [nn.Conv2d(out_channels, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=True)]
        super(NLayerDiscriminator, self).__init__(*layers)


if __name__ == "__main__":
    print(torch.__version__)
    g = NLayerDiscriminator(3)
    input = torch.rand((1, 3, 512, 512))
    print(g(input).size())

    input = torch.rand((1, 3, 511, 511))
    print(g(input).size())
