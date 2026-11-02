import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection (Lin et al., ICCV 2017).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where alpha_t = alpha for positive class (targets == 1)
    and (1 - alpha) for negative class (targets == 0).
    """
    def __init__(self, alpha=0.5, gamma=2.0, num_classes=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        # Class-dependent alpha balancing
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = alpha_t * ((1.0 - pt) ** self.gamma) * bce
        return focal.mean()


def build_loss(num_classes=1):
    return FocalLoss(num_classes=num_classes)
