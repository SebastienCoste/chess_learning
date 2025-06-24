import torch
import torch.nn as nn
import torch.nn.functional as F

class MishActivation(nn.Module):
    """
    Mish activation function implementation.
    Mish(x) = x * tanh(softplus(x))
    """

    def __init__(self):
        super().__init__()
        # Use built-in Mish if available (PyTorch 1.9+)
        if hasattr(F, 'mish'):
            self.act = F.mish
        else:
            self.act = self._mish_implementation

    def _mish_implementation(self, x):
        return x * torch.tanh(F.softplus(x))

    def forward(self, x):
        return self.act(x)