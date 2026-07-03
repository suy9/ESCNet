import torch
import torch.nn as nn

from models.modules.blocks import DSConv, FeatureReduce, bilinear_resize
from models.modules.utils import SA


class AETP(nn.Module):
    """Adaptive Edge-aware Transformer Pyramid — predicts object boundaries."""

    def __init__(self, channels: int):
        super().__init__()
        self.inject_top = nn.ModuleList(
            FeatureReduce(channels, channels) for _ in range(3)
        )
        # These paths carry no edge guidance, so cheap DSConv replaces DeformableConv.
        self.refine = nn.ModuleList(
            DSConv(channels, channels, kernel_size=3) for _ in range(3)
        )
        self.attn = nn.ModuleList(
            SA(channels, ffn_expansion_factor=1) for _ in range(3)
        )

        self.merge_mid = nn.Conv2d(2 * channels, channels, 3, 1, 1)
        self.edge_fusion = nn.Conv2d(3 * channels, channels, 3, 1, 1)
        self.out = nn.Conv2d(channels, 1, 1)

    def forward(self, features):
        image, x1, x2, x3, x4 = features

        x3 = self.inject_top[0](x3, x4)
        x2 = self.inject_top[1](x2, x4)
        x1 = self.inject_top[2](x1, x4)

        x4 = self.refine[0](bilinear_resize(x4, x3.shape[2:]))
        fused_l3 = x3 + x4

        x3 = self.refine[1](bilinear_resize(x3, x2.shape[2:]))
        fused_l2 = x2 + x3
        fused_l3 = bilinear_resize(fused_l3, x2.shape[2:])

        x2 = self.refine[2](bilinear_resize(x2, x1.shape[2:]))
        fused_l1 = x1 + x2

        fused_l3 = self.attn[0](fused_l3)
        side_l2 = torch.cat((fused_l2, fused_l3), dim=1)

        fused_l2 = self.attn[1](bilinear_resize(fused_l2, x1.shape[2:]))
        side_l1 = torch.cat((fused_l1, fused_l2), dim=1)

        side_l2 = bilinear_resize(side_l2, x1.shape[2:])
        side_l2 = self.attn[2](self.merge_mid(side_l2))

        edge_feat = self.edge_fusion(torch.cat((side_l2, side_l1), dim=1))
        edge_feat = bilinear_resize(edge_feat, image.shape[2:])
        return self.out(edge_feat)
