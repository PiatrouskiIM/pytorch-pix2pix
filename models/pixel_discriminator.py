import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union


class PixelDiscriminator(nn.Sequential):

    def __init__(self, in_channels, multiplier=64, norm_layer: Optional[Callable[..., nn.Module]] = None,):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        super(PixelDiscriminator, self).__init__(
            nn.Conv2d(in_channels, multiplier, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(multiplier, multiplier * 2, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(multiplier * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(multiplier * 2, 1, kernel_size=1, stride=1, padding=0, bias=False))


if __name__ == "__main__":
    print(torch.__version__)
    g = PixelDiscriminator(3)
    input = torch.rand((1, 3, 512, 512))
    print(g(input).size())

    input = torch.rand((1, 3, 511, 511))
    print(g(input).size())
