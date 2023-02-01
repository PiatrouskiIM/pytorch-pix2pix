import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock as ResnetBasicBlock
from typing import Any, Callable, List, Optional, Type, Union
import numpy as np


# diff
# 1. padding mode reflect -> zero
# 2. torchvision implementation of ResnetBasicBlock contains activation at the end
# 3. use bias set to false (Note, bias is only suitable for nn.InstanceNorm2d with affine=False,
# nn.BatchNorm2d and nn.InstanceNorm2d with affine=True will neglect effect of bias,
# (see, https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
# Note: stargan version of the resnet generator use kernel_size 4 and no output padding for sampling_kwargs


class ResnetGenerator(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 multiplier=64,
                 num_hidden_blocks=6,
                 num_resolution_halving=2,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, ):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        channels = multiplier * 2 ** num_resolution_halving
        layers = [ResnetBasicBlock(inplanes=channels, planes=channels) for _ in range(num_hidden_blocks)]

        sampling_kwargs = dict(kernel_size=3, stride=2, padding=1, bias=False)
        for channels in multiplier * 2 ** (num_resolution_halving - np.arange(num_resolution_halving)):
            layers = [
                         nn.Conv2d(channels // 2, channels, **sampling_kwargs),
                         norm_layer(channels),
                         nn.ReLU(inplace=True)
                     ] + layers + [
                         nn.ConvTranspose2d(channels, channels // 2, output_padding=1, **sampling_kwargs),
                         norm_layer(channels // 2),
                         nn.ReLU(inplace=True)
                     ]

        into = [
            nn.Conv2d(in_channels, multiplier, kernel_size=7, stride=1, padding=3, bias=False),
            norm_layer(multiplier),
            nn.ReLU(inplace=True)
        ]
        outro = [
            nn.Conv2d(multiplier, out_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        ]

        super(ResnetGenerator, self).__init__(*into, *layers, *outro)


if __name__ == "__main__":
    print(torch.__version__)
    g = ResnetGenerator(3, 3)
    input = torch.rand((1, 3, 511, 511))
    print(g(input).size())
