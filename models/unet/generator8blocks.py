import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union

from .up_block import UpBlock
from .down_block import DownBlock


class Generator8Blocks(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 multiplier: int = 64,
                 use_dropout: bool = False,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, ):

        super(Generator8Blocks, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.intro = nn.Conv2d(in_channels, multiplier, kernel_size=4, stride=2, padding=1, bias=False)
        self.down_stack = nn.ModuleList([
            DownBlock(multiplier * 2 ** 0, multiplier * 2 ** 1, norm_layer=norm_layer),
            DownBlock(multiplier * 2 ** 1, multiplier * 2 ** 2, norm_layer=norm_layer),
            DownBlock(multiplier * 2 ** 2, multiplier * 2 ** 3, norm_layer=norm_layer),

            DownBlock(multiplier * 2 ** 3, multiplier * 2 ** 3, norm_layer=norm_layer),
            DownBlock(multiplier * 2 ** 3, multiplier * 2 ** 3, norm_layer=norm_layer),
            DownBlock(multiplier * 2 ** 3, multiplier * 2 ** 3, norm_layer=norm_layer),

            DownBlock(multiplier * 2 ** 3, multiplier * 2 ** 3, norm_layer=None),
        ])

        self.up_stack = nn.ModuleList([
            UpBlock(multiplier * 2 ** 3, multiplier * 2 ** 3, use_dropout=False, norm_layer=norm_layer),

            UpBlock(multiplier * 2 ** 4, multiplier * 2 ** 3, use_dropout=use_dropout, norm_layer=norm_layer),
            UpBlock(multiplier * 2 ** 4, multiplier * 2 ** 3, use_dropout=use_dropout, norm_layer=norm_layer),
            UpBlock(multiplier * 2 ** 4, multiplier * 2 ** 3, use_dropout=use_dropout, norm_layer=norm_layer),

            UpBlock(multiplier * 2 ** 4, multiplier * 2 ** 2, use_dropout=False, norm_layer=norm_layer),
            UpBlock(multiplier * 2 ** 3, multiplier * 2 ** 1, use_dropout=False, norm_layer=norm_layer),
            UpBlock(multiplier * 2 ** 2, multiplier * 2 ** 0, use_dropout=False, norm_layer=norm_layer),
        ])
        self.outro = nn.ConvTranspose2d(multiplier * 2 ** 1,
                                        out_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias=False)

    def forward(self, x):
        x = self.intro(x)
        scips = []
        for module in self.down_stack:
            scips.append(x)
            x = module(x)
        for module, skip in zip(self.up_stack, scips[::-1]):
            x = module(x)
            x = torch.cat([x, skip], dim=1)
        x = self.outro(x)
        return torch.tanh(x)
