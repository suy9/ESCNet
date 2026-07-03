import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import laplacian
from torchvision.transforms.functional import rgb_to_grayscale

from models.modules.blocks import bilinear_resize
from models.modules.utils import SA


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x) * x
        return self.spatial_attention(x) * x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pooled = self.avg_pool(x).squeeze(-1).squeeze(-1), self.max_pool(x).squeeze(
            -1
        ).squeeze(-1)
        weights = self.sigmoid(self.mlp(pooled[0]) + self.mlp(pooled[1]))
        return weights.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        weights = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return weights


class MTA(nn.Module):
    """Multi-source Token Attention — fuses mask, edge, and texture cues."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.edge_weight = nn.Parameter(torch.tensor(1.0))
        self.texture_weight = nn.Parameter(torch.tensor(1.0))
        self.fusion = nn.Conv2d(in_channels * 3, out_channels, 3, 1, 1)
        self.mask_attn = SA(in_channels)
        self.edge_attn = SA(in_channels)
        self.texture_attn = SA(in_channels)
        self.refine = CBAM(out_channels)

    def forward(self, x, coarse_mask, edge, image):
        spatial_size = x.shape[2:]
        mask_guided = self.mask_attn(
            x, bilinear_resize(coarse_mask.detach(), spatial_size)
        )
        edge_guided = self.edge_attn(
            x, bilinear_resize(edge.detach(), spatial_size)
        ) * self.edge_weight

        grayscale = rgb_to_grayscale(image)
        texture = laplacian(grayscale, kernel_size=5, normalized=True)
        texture = bilinear_resize(texture, spatial_size)
        texture_guided = self.texture_attn(x, texture) * self.texture_weight

        fused = self.fusion(torch.cat((mask_guided, edge_guided, texture_guided), dim=1))
        return self.refine(fused)
