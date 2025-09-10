import pdb
from models.modules.utils import DeformableConv, image2patches
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from models.modules.MFMM import MTA


class ConvsByConvs(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inter_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


class FEM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, edge=False):
        super(FEM, self).__init__()
        self.dwconv = DeformableConv(
            in_channels, out_channels, 3, 1, 1, edge=edge
        )  
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 5, 1, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 7, 1, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(out_channels * 4, out_channels, 1, 1, 0)

    def forward(self, x, edge=None):
        if edge is not None:
            dw = self.dwconv(x, edge)
        else:
            dw = self.dwconv(x)

        x1 = self.conv1x1(dw)
        x3 = self.conv3x3(dw + x1)
        x5 = self.conv5x5(dw + x3)
        x7 = self.conv7x7(dw + x5)

        concat_features = torch.cat([x1, x3, x5, x7], dim=1)

        fused_features = self.final_conv(concat_features)

        return fused_features + dw


class Decoder(nn.Module):
    def __init__(self, config, in_channel):
        super(Decoder, self).__init__()
        self.config = config

        self.ipt_blk5 = ConvsByConvs(  # 将patch做下通道卷积操作
            2**10 * 3,
            in_channel // 4,
        )
        self.ipt_blk4 = ConvsByConvs(
            2**8 * 3,
            in_channel // 4,
        )
        self.ipt_blk3 = ConvsByConvs(
            2**6 * 3,
            in_channel // 4,
        )
        self.ipt_blk2 = ConvsByConvs(
            2**4 * 3,
            in_channel // 4,
        )
        self.decoder_block4 = ConvsByConvs(in_channel, in_channel)

        self.decoder_block3 = MTA(
            in_channel,
            in_channel,
        )

        self.decoder_block2 = MTA(
            in_channel,
            in_channel,
        )

        self.decoder_block1 = MTA(
            in_channel,
            in_channel,
        )

        self.conv_mask_4 = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, 1, 0),
            # nn.BatchNorm2d(ic),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(ic, 1, 1, 1, 0),
        )
        self.conv_mask_3 = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, 1, 0),
            # nn.BatchNorm2d(ic),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(ic, 1, 1, 1, 0),
        )
        self.conv_mask_2 = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, 1, 0),
            # nn.BatchNorm2d(ic),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(ic, 1, 1, 1, 0),
        )

        self.tra_fr = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channel, in_channel // 2, 1, 1, 0),
            nn.BatchNorm2d(in_channel // 2),
            nn.ReLU(inplace=True),
        )
        self.predictor_fr = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channel // 2, in_channel // 2, 3, 1, 1),
            nn.BatchNorm2d(in_channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 2, 1, 1, 1, 0),
        )

        self.De_conv4 = FEM(in_channel + in_channel // 4, in_channel, edge=True)
        self.De_conv3 = FEM(in_channel + in_channel // 4, in_channel, edge=True)
        self.De_conv2 = FEM(in_channel + in_channel // 4, in_channel, edge=True)
        self.De_conv1 = FEM(in_channel + in_channel // 4, in_channel, edge=True)

    def forward(self, features, edge):
        x, x1, x2, x3, x4 = features

        patches_batch = image2patches(
            x, patch_ref=x4, transformation="b c (hg h) (wg w) -> b (c hg wg) h w"
        )
        x4 = torch.cat(
            (
                x4,
                self.ipt_blk5(
                    F.interpolate(
                        patches_batch,
                        size=x4.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                ),
            ),
            1,
        )
        x4 = self.De_conv4(
            x4,
            edge=F.interpolate(
                edge, size=x4.shape[2:], mode="bilinear", align_corners=False
            ),
        )

        patches_batch = image2patches(
            x, patch_ref=x3, transformation="b c (hg h) (wg w) -> b (c hg wg) h w"
        )
        x3 = torch.cat(
            (
                x3,
                self.ipt_blk4(
                    F.interpolate(
                        patches_batch,
                        size=x3.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                ),
            ),
            1,
        )
        x3 = self.De_conv3(
            x3,
            edge=F.interpolate(
                edge, size=x3.shape[2:], mode="bilinear", align_corners=False
            ),
        )

        patches_batch = image2patches(
            x, patch_ref=x2, transformation="b c (hg h) (wg w) -> b (c hg wg) h w"
        )
        x2 = torch.cat(
            (
                x2,
                self.ipt_blk3(
                    F.interpolate(
                        patches_batch,
                        size=x2.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                ),
            ),
            1,
        )
        x2 = self.De_conv2(
            x2,
            edge=F.interpolate(
                edge, size=x2.shape[2:], mode="bilinear", align_corners=False
            ),
        )

        patches_batch = image2patches(
            x, patch_ref=x1, transformation="b c (hg h) (wg w) -> b (c hg wg) h w"
        )
        x1 = torch.cat(
            (
                x1,
                self.ipt_blk2(
                    F.interpolate(
                        patches_batch,
                        size=x1.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                ),
            ),
            1,
        )
        x1 = self.De_conv1(
            x1,
            edge=F.interpolate(
                edge, size=x1.shape[2:], mode="bilinear", align_corners=False
            ),
        )

        out_mask = []

        p4 = self.decoder_block4(x4)

        m4 = self.conv_mask_4(p4)

        _p4 = F.interpolate(p4, size=x3.shape[2:], mode="bilinear", align_corners=False)
        _p3 = _p4 + x3

        p3 = self.decoder_block3(_p3, m4, edge, x)
        m3 = self.conv_mask_3(p3)
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode="bilinear", align_corners=False)
        _p2 = _p3 + x2

        p2 = self.decoder_block2(_p2, m3, edge, x)
        m2 = self.conv_mask_2(p2)
        _p2 = F.interpolate(p2, size=x1.shape[2:], mode="bilinear", align_corners=False)
        _p1 = _p2 + x1

        p1 = self.decoder_block1(_p1, m2, edge, x)
        m1 = self.predictor_fr(self.tra_fr(p1))

        out_mask.append(m4)
        out_mask.append(m3)
        out_mask.append(m2)
        out_mask.append(m1)

        return out_mask
