from models.modules.Decoder import Decoder
from models.modules.AETP import AETP
from models.modules.Encoder import Encoder
import torch.nn as nn


# @torch.compile()
class ESCNet(nn.Module):
    def __init__(self, config, pretrained=True):
        super(ESCNet, self).__init__()
        self.channels = config.lateral_channels
        inter_channel = 128

        self.encoder = Encoder(config, pretrained)
        self.decoder = Decoder(config,inter_channel)
        self.enhanced = AETP(inter_channel)

        self.asa4 = nn.Sequential(
            nn.Conv2d(self.channels[0], inter_channel, 1, 1, 0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
        )
        self.asa3 = nn.Sequential(
            nn.Conv2d(self.channels[1], inter_channel, 1, 1, 0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
        )
        self.asa2 = nn.Sequential(
            nn.Conv2d(self.channels[2], inter_channel, 1, 1, 0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
        )
        self.asa1 = nn.Sequential(
            nn.Conv2d(self.channels[3], inter_channel, 1, 1, 0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        ########## Encoder ##########
        (x1, x2, x3, x4) = self.encoder(x)  # x1,x2,x3,x1+x2+x3+x4
        x4 = self.asa4(x4)
        x3 = self.asa3(x3)
        x2 = self.asa2(x2)
        x1 = self.asa1(x1)
        features = [x, x1, x2, x3, x4]

        ########## Decoder ##########
        out_edge = self.enhanced(features)  # logits

        out_mask = self.decoder(features, out_edge.sigmoid())
        return out_edge, out_mask
