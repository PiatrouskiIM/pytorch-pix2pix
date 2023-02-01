import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union


class UpBlock(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_dropout=False,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 bias=False,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 **conv_kwargs):

        layers = [
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=bias,
                               **conv_kwargs)
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if use_dropout:
            layers += [nn.Dropout(0.5)]
        super(UpBlock, self).__init__(*layers)
