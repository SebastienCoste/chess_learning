import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Focal Loss addresses class imbalance by down-weighting well-classified examples . This is particularly useful for chess move prediction where some moves are much more common than others.
"""
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

#For more severe class imbalance
class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.9999):
        super(ClassBalancedLoss, self).__init__()
        effective_num = 1.0 - torch.pow(beta, torch.tensor(samples_per_class))
        weights = (1.0 - beta) / effective_num
        self.weights = weights / weights.sum() * len(samples_per_class)

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.weights.to(inputs.device))
