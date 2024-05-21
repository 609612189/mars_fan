import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import  trunc_normal_
from ultralytics.nn.modules import C3, C2f, Bottleneck, Conv
from ultralytics.nn.Addmodules.DAttention2 import DAttention


__all__ = ['C2f_DWRSeg_DAttention']

class DWR(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv_3x3 = Conv(dim, dim // 2, 3)

        self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1)
        self.conv_3x3_d3 = Conv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = Conv(dim // 2, dim // 2, 3, d=5)

        self.conv_1x1 = Conv(dim * 2, dim, k=1)

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out


class DWRSeg_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, dilation=1):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, k=1)
        self.dcnv3 = DWR(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.dcnv3(x)
        x = self.gelu(self.bn(x))
        return x


class Bottleneck_DWRSeg_DAttention(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, fmapsize, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DWRSeg_Conv(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2
        self.attention = DAttention(c2, fmapsize)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))

class C2f_DWRSeg_DAttention(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, fmapsize=None, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            Bottleneck_DWRSeg_DAttention(self.c, self.c, fmapsize, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
