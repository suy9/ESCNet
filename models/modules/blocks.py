"""Reusable building blocks shared across ESCNet modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def bilinear_resize(x: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 1,
    padding: int = 0,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DSConv(nn.Module):
    """Depthwise-separable conv (depthwise KxK + pointwise 1x1) with BN + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            1,
            kernel_size // 2,
            groups=in_channels,
            bias=False,
        )
        self.pw = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class ConvBottleneck(nn.Module):
    """1x1 projection followed by 3x3 convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inter_channels: int = 64,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv1(x)))
        return self.conv2(x)


class ChannelAlign(nn.Module):
    """Project backbone features to a common channel width (ASA block)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = conv_bn_relu(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FeatureReduce(nn.Module):
    """Fuse concatenated multi-scale features back to a single resolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, feat: torch.Tensor, top_feat: torch.Tensor) -> torch.Tensor:
        top_resized = bilinear_resize(top_feat, feat.shape[2:])
        return self.reduce(torch.cat([feat, top_resized], dim=1))
