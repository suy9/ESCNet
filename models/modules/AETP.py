from models.modules.utils import SA, DeformableConv
import torch
import torch.nn as nn
import torch.nn.functional as F



class AETP(nn.Module):
    def __init__(self, in_channel):
        super(AETP, self).__init__()
        self.reduce3 = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
        )
        self.reduce2 = nn.Sequential(
            nn.Conv2d(in_channel*2, in_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
        )
        self.reduce1 = nn.Sequential(
            nn.Conv2d(in_channel*2, in_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
        )
        self.conv4_1 = DeformableConv(in_channel, in_channel, 3, 1, 1,edge=False)
        self.conv3_1 = DeformableConv(in_channel, in_channel, 3, 1, 1,edge=False)
        self.conv2_1 = DeformableConv(in_channel, in_channel, 3, 1, 1,edge=False)

        self.sa_f3 = SA(
            in_channel,
            ffn_expansion_factor=1
        )
        self.sa_f2 = SA(
            in_channel,
            ffn_expansion_factor=1
        )
        self.sa_s2 = SA(
            in_channel,
            ffn_expansion_factor=1
        )
        self.conv_s2 = nn.Conv2d(2 * in_channel, in_channel, 3, 1, 1)
        # self.edge = SA(
        #     in_channel,
        # )
        # self.edge = DeformableConv(
        #     in_channel,
        #     in_channel,
        #     3,1,1
        # )
        self.edge_conv = nn.Conv2d(3 * in_channel, in_channel, 3, 1, 1)

        self.out = nn.Conv2d(in_channel, 1, 1, 1, 0)

    def forward(self, features):
        x, x1, x2, x3, x4 = features  # encoder的输出

        x3 = torch.cat(
            (
                x3,
                F.interpolate(x4, size=x3.shape[2:], mode="bilinear", align_corners=False),
            ),
            dim=1,
        )
        x3 = self.reduce3(x3)
        x2 = torch.cat(
            (
                x2,
                F.interpolate(x4, size=x2.shape[2:], mode="bilinear", align_corners=False),
            ),
            dim=1,
        )
        x2 = self.reduce2(x2)
        x1 = torch.cat(
            (
                x1,
                F.interpolate(x4, size=x1.shape[2:], mode="bilinear", align_corners=False),
            ),
            dim=1,
        )
        x1 = self.reduce1(x1)

        x4 = F.interpolate(x4, size=x3.size()[-2:], mode="bilinear", align_corners=False)
        x4 = self.conv4_1(x4)
        f3 = x3 + x4

        x3 = F.interpolate(x3, size=x2.size()[-2:], mode="bilinear", align_corners=False)
        x3 = self.conv3_1(x3)
        f2 = x2 + x3

        f3 = F.interpolate(f3, size=x2.size()[-2:], mode="bilinear", align_corners=False)

        x2 = F.interpolate(x2, size=x1.size()[-2:], mode="bilinear", align_corners=False)
        x2 = self.conv2_1(x2)
        f1 = x1 + x2

        f3 = self.sa_f3(f3)

        s2 = torch.cat((f2, f3), dim=1)

        f2 = F.interpolate(f2, size=x1.size()[-2:], mode="bilinear", align_corners=False)
        f2 = self.sa_f2(f2)

        s1 = torch.cat((f1, f2), dim=1)

        s2 = F.interpolate(s2, size=x1.size()[-2:], mode="bilinear", align_corners=False)

        s2 = self.conv_s2(s2)
        s2 = self.sa_s2(s2)
        
        edge = torch.cat((s2, s1), dim=1)
        edge = self.edge_conv(edge)

        edge = F.interpolate(edge, size=x.size()[-2:], mode="bilinear", align_corners=False)
        out = self.out(edge)
        return out 
