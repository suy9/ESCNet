import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from models.backbones.build_backbone import build_backbone


def feature_concat(feature, feature_resized, size, dim):
    return torch.cat(
        [
            feature,
            F.interpolate(
                feature_resized, size=size, mode="bilinear", align_corners=False
            ),
        ],
        dim=dim,
    )


class Encoder(nn.Module):
    def __init__(self, config, bb_pretrained=True):
        super(Encoder, self).__init__()
        self.config = config
        self.backbone = build_backbone(self.config, self.config.backbone, pretrained=bb_pretrained)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)

        # x4 = torch.cat(
        #     (
        #         F.interpolate(
        #             x1, size=x4.shape[2:], mode="bilinear", align_corners=False
        #         ),
        #         F.interpolate(
        #             x2, size=x4.shape[2:], mode="bilinear", align_corners=False
        #         ),
        #         F.interpolate(
        #             x3, size=x4.shape[2:], mode="bilinear", align_corners=False
        #         ),
        #         x4,
        #     ),
        #     dim=1,
        # )
        return x1, x2, x3, x4
