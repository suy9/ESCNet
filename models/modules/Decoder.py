import torch
import torch.nn as nn

from models.modules.MFMM import MTA
from models.modules.blocks import ConvBottleneck, DSConv, bilinear_resize
from models.modules.utils import DeformableConv, image2patches

PATCH_TRANSFORM = "b c (hg h) (wg w) -> b (c hg wg) h w"
PATCH_CHANNELS = [2**10 * 3, 2**8 * 3, 2**6 * 3, 2**4 * 3]


class FEM(nn.Module):
    """Feature Enhancement Module with multi-scale convolutions and optional edge guidance."""

    def __init__(self, in_channels: int, out_channels: int, edge: bool = False):
        super().__init__()
        self.dwconv = DeformableConv(in_channels, out_channels, 3, 1, 1, edge=edge)
        # Depthwise-separable branches keep the large receptive fields cheap.
        self.branches = nn.ModuleList(
            [DSConv(out_channels, out_channels, kernel_size=k) for k in (1, 3, 5, 7)]
        )
        self.fuse = nn.Conv2d(out_channels * 4, out_channels, 1, 1, 0)

    def forward(self, x, edge=None):
        dw = self.dwconv(x, edge) if edge is not None else self.dwconv(x)

        x1 = self.branches[0](dw)
        x3 = self.branches[1](dw + x1)
        x5 = self.branches[2](dw + x3)
        x7 = self.branches[3](dw + x5)
        return self.fuse(torch.cat([x1, x3, x5, x7], dim=1)) + dw


class Decoder(nn.Module):
    """Semantic decoder with patch injection, edge-guided FEM, and multi-scale supervision."""

    def __init__(self, config, channels: int):
        super().__init__()
        patch_dim = channels // 4
        fem_in = channels + patch_dim

        self.patch_projs = nn.ModuleList(
            ConvBottleneck(patch_ch, patch_dim) for patch_ch in PATCH_CHANNELS
        )
        self.fem_layers = nn.ModuleList(
            FEM(fem_in, channels, edge=True) for _ in PATCH_CHANNELS
        )

        self.coarse_head = ConvBottleneck(channels, channels)
        self.decode_blocks = nn.ModuleList(MTA(channels, channels) for _ in range(3))
        self.mask_heads = nn.ModuleList(nn.Conv2d(channels, 1, 1) for _ in range(3))

        self.upsample_refine = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
        )
        self.final_mask_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels // 2, channels // 2, 3, 1, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
        )

    def _fuse_patches(self, image, feature, patch_proj, fem, edge):
        patches = image2patches(
            image, patch_ref=feature, transformation=PATCH_TRANSFORM
        )
        patch_feat = patch_proj(bilinear_resize(patches, feature.shape[2:]))
        fused = torch.cat([feature, patch_feat], dim=1)
        return fem(fused, edge=bilinear_resize(edge, feature.shape[2:]))

    def _enhance_features(self, image, feats, edge):
        # feats: [x1, x2, x3, x4] shallow -> deep; fuse deep -> shallow.
        levels = list(reversed(feats))
        return [
            self._fuse_patches(image, feat, proj, fem, edge)
            for feat, proj, fem in zip(levels, self.patch_projs, self.fem_layers)
        ]

    def forward(self, features, edge):
        image, x1, x2, x3, x4 = features
        x4, x3, x2, x1 = self._enhance_features(image, [x1, x2, x3, x4], edge)

        feat = self.coarse_head(x4)
        prev_mask = self.mask_heads[0](feat)
        masks = [prev_mask]

        skips = (x3, x2, x1)
        for i, skip in enumerate(skips):
            feat = self.decode_blocks[i](
                bilinear_resize(feat, skip.shape[2:]) + skip, prev_mask, edge, image
            )
            if i < len(skips) - 1:
                prev_mask = self.mask_heads[i + 1](feat)
                masks.append(prev_mask)
            else:
                masks.append(self.final_mask_head(self.upsample_refine(feat)))

        return masks
