import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, num_classes=4):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        loss_ce = self.ce(inputs, targets)

        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = probs.sum(dims) + targets_one_hot.sum(dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss_dice = 1.0 - dice_per_class.mean()

        return 0.5 * loss_ce + 0.5 * loss_dice


def build_ce_loss(num_classes=4):
    return nn.CrossEntropyLoss()


def build_ce_dice_loss(num_classes=4):
    return CrossEntropyDiceLoss(num_classes=num_classes)

