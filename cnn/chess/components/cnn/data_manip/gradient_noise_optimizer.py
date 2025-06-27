import torch
from torch.optim import Optimizer


class GradientNoiseOptimizer(Optimizer):
    def __init__(self, optimizer, noise_std=0.01, decay=0.55):
        # Register the base optimizer
        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups  # Required for scheduler compatibility
        self.state = optimizer.state

        # Noise parameters
        self.noise_std = noise_std
        self.decay = decay
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        current_noise = self.noise_std / (1 + self.step_count) ** self.decay

        # Add gradient noise
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * current_noise
                    param.grad.add_(noise)

        # Delegate to base optimizer
        return self.optimizer.step(closure)

    def zero_grad(self, set_to_none=True):
        return self.optimizer.zero_grad(set_to_none)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)
