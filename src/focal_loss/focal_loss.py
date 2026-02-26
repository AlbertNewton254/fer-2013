import torch
import torch.nn as nn
import torch.nn.functional as F
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

        alpha_t = self.alpha
        if torch.is_tensor(alpha_t):
            alpha_t = alpha_t.to(device=inputs.device, dtype=inputs.dtype)
            if alpha_t.dim() != 1:
                raise ValueError("alpha tensor must be 1D with per-class weights")
            one_hot_targets = F.one_hot(targets, num_classes=inputs.size(1)).to(inputs.dtype)
            alpha_t = (alpha_t.unsqueeze(0) * one_hot_targets).sum(dim=1)
        else:
            alpha_t = torch.tensor(alpha_t, device=inputs.device, dtype=inputs.dtype)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss