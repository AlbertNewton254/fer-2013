import torch
import torch.nn as nn
from src.config import get_config


FOCAL_LOSS_CONFIG = get_config()["loss"]["focal"]

class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=FOCAL_LOSS_CONFIG["alpha"],
        gamma=FOCAL_LOSS_CONFIG["gamma"],
        reduction=FOCAL_LOSS_CONFIG["reduction"],
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss