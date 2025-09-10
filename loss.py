import torch
from torch import nn
import torch.nn.functional as F



class EdgeDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(EdgeDiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, target, valid_mask=None):
        pred = torch.sigmoid(logits)

        if valid_mask is None:
            valid_mask = torch.ones_like(target)

        # 展平张量 (B, H*W)
        pred = pred.view(pred.shape[0], -1)
        target = target.view(target.shape[0], -1)
        valid_mask = valid_mask.view(valid_mask.shape[0], -1)

        intersection = torch.sum(pred * target * valid_mask, dim=1)
        total = torch.sum((pred + target) * valid_mask, dim=1)
        dice = (2 * intersection + self.epsilon) / (total + self.epsilon)
        loss = 1 - dice
        return loss.mean()


class EdgeDiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, epsilon=1e-6):
        """
        alpha: Dice Loss 权重（越大越关注边缘）
        gamma: Focal Loss 聚焦难例的程度
        """
        super().__init__()
        self.dice = EdgeDiceLoss(epsilon)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target, valid_mask=None):
        dice_loss = self.dice(logits, target, valid_mask)

        # Focal Loss（直接基于 Logits 计算）
        pred = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        pt = pred * target + (1 - pred) * (1 - target)  # 计算 pt = p_t
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = (focal_weight * bce_loss).mean()

        # 加权结合（确保权重和为 1）
        return self.alpha * dice_loss + (1 - self.alpha) * focal_loss


class WBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(WBCEWithLogitsLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction="none"
        )  # 不进行平均，而是逐像素计算损失

    def forward(self, pred, target, weight=None):
        b = pred.shape[0]
        # 计算原始的 BCE 损失
        loss = self.bce_loss(pred, target)

        # 如果有权重，应用权重
        if weight is not None:
            # 逐像素加权损失
            loss = loss * weight

        # 返回平均损失
        return loss.mean()



class StructureLossWithWeight(torch.nn.Module):

    def __init__(self, epision=1e-6):
        super().__init__()
        self.epision = epision

    def forward(self, pred, target):
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
        )
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * target) * weit).sum(dim=(2, 3))
        union = ((pred + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + self.epision) / (union - inter + self.epision)

        return (wbce + wiou).mean()


class StructureLoss(torch.nn.Module):

    def __init__(self, epision=1e-6):
        super().__init__()
        self.epision = epision

    def forward(self, pred, target):
        # pdb.set_trace()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        bce = bce.mean(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3))
        iou = 1 - (inter + self.epision) / (union - inter + self.epision)

        return (bce + iou).mean()
