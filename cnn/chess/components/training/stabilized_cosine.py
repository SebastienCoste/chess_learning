import torch
import math

class StabilizedCosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Gradual restart transition
        transition_ratio = min(1.0, (self.last_epoch - self.T_cur) / max(1, self.T_0 // 4))

        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(transition_ratio * math.pi)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.T_cur = self.T_cur + 1
        if self.T_cur >= self.T_0:
            self.T_cur = 0
            self.T_0 = int(self.T_0 * self.T_mult)
        super().step(epoch)
