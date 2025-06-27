import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MishActivation(nn.Module):
    """
    Mish activation function implementation.
    Mish(x) = x * tanh(softplus(x))
    """
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.mish(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str