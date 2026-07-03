import torch.nn as nn

from models.backbones.build_backbone import build_backbone


class Encoder(nn.Module):
    """Multi-scale feature extractor built on a configurable backbone."""

    def __init__(self, config, bb_pretrained=True):
        super().__init__()
        self.backbone = build_backbone(
            config, config.backbone, pretrained=bb_pretrained
        )

    def forward(self, x):
        return self.backbone(x)
