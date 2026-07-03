import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, target, valid_mask=None):
        pred = torch.sigmoid(logits)
        valid_mask = valid_mask if valid_mask is not None else torch.ones_like(target)

        pred = pred.flatten(1)
        target = target.flatten(1)
        valid_mask = valid_mask.flatten(1)

        intersection = (pred * target * valid_mask).sum(dim=1)
        total = ((pred + target) * valid_mask).sum(dim=1)
        dice = (2 * intersection + self.epsilon) / (total + self.epsilon)
        return (1 - dice).mean()


class EdgeDiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, epsilon=1e-6):
        super().__init__()
        self.dice = EdgeDiceLoss(epsilon)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target, valid_mask=None):
        dice_loss = self.dice(logits, target, valid_mask)

        pred = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = pred * target + (1 - pred) * (1 - target)
        focal_loss = ((1 - pt) ** self.gamma * bce_loss).mean()

        return self.alpha * dice_loss + (1 - self.alpha) * focal_loss


class WBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target, weight=None):
        loss = self.bce_loss(pred, target)
        if weight is not None:
            loss = loss * weight
        return loss.mean()


class StructureLossWithWeight(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        weight = 1 + 5 * torch.abs(
            F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
        )
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = (pred * target * weight).sum(dim=(2, 3))
        union = ((pred + target) * weight).sum(dim=(2, 3))
        wiou = 1 - (inter + self.epsilon) / (union - inter + self.epsilon)
        return (wbce + wiou).mean()


class StructureLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        bce = bce.mean(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3))
        iou = 1 - (inter + self.epsilon) / (union - inter + self.epsilon)
        return (bce + iou).mean()
