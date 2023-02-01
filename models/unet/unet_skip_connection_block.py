import torch
import torch.nn as nn
import functools
from typing import Any, Callable, List, Optional, Type, Union

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 input_nc=None,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 use_dropout=False,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,):
        super(UnetSkipConnectionBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc


        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            # up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2,
                                        outer_nc,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            model = down + [submodule] + up

            if use_dropout:
                model += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], dim=1)
