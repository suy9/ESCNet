import torch.nn as nn

from models.modules.AETP import AETP
from models.modules.Decoder import Decoder
from models.modules.Encoder import Encoder
from models.modules.blocks import ChannelAlign

INTER_CHANNELS = 128


class ESCNet(nn.Module):
    """Edge-Semantic Collaborative Network for camouflaged object detection."""

    def __init__(self, config, pretrained=True):
        super().__init__()
        self.encoder = Encoder(config, pretrained)
        self.edge_branch = AETP(INTER_CHANNELS)
        self.mask_decoder = Decoder(config, INTER_CHANNELS)

        # Deepest -> shallowest backbone levels.
        self.channel_align = nn.ModuleList(
            ChannelAlign(ch, INTER_CHANNELS) for ch in config.lateral_channels
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)

        # lateral_channels is ordered deep -> shallow (x4, x3, x2, x1).
        x4 = self.channel_align[0](x4)
        x3 = self.channel_align[1](x3)
        x2 = self.channel_align[2](x2)
        x1 = self.channel_align[3](x1)

        features = [x, x1, x2, x3, x4]
        edge_logits = self.edge_branch(features)
        mask_preds = self.mask_decoder(features, edge_logits.sigmoid())
        return edge_logits, mask_preds
