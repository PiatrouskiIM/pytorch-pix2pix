import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union


class DownBlock(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 bias=False,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 **conv_kwargs):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        layers = [
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias,
                      **conv_kwargs)
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        super(DownBlock, self).__init__(*layers)
