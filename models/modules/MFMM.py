from models.modules.utils import SA
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import laplacian
from torchvision.transforms.functional import rgb_to_grayscale


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):

        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.mlp(self.max_pool(x).squeeze(-1).squeeze(-1))

        channel_weights = self.sigmoid(avg_out + max_out)

        return channel_weights.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.sigmoid(self.conv(combined))
        return spatial_weights


class MTA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super(MTA, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 3, 1, 1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.sa1 = SA(in_channels)
        self.sa2 = SA(in_channels)
        self.sa3 = SA(in_channels)

        self.cbam = CBAM(out_channels)

    def forward(self, x, pred, edge, I):
        edge_d = edge.detach()
        mask_d = pred.detach()
        edge_d = F.interpolate(
            edge_d, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        mask_d = F.interpolate(
            mask_d, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        x1 = self.sa1(x, mask_d)

        x2 = self.sa2(x, edge_d) * self.alpha

        grayscale_img = rgb_to_grayscale(I)
        laplace = laplacian(grayscale_img, kernel_size=5, normalized=True)

        laplace = F.interpolate(
            laplace, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        x3 = self.sa3(x, laplace) * self.beta

        zz = torch.cat((x1, x2, x3), dim=1)
        out = self.conv1(zz)

        out_cbam = self.cbam(out)

        return out_cbam
