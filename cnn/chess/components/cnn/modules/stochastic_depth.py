import torch
import torch.nn as nn

"""
Stochastic depth randomly drops entire residual blocks during training, acting like dropout but at the block level . This prevents co-adaptation of layers and improves generalization, especially in deep networks.
"""
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.1):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x

        # Create binary mask for the entire batch
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()  # binarize

        # Scale the kept activations
        return x * random_tensor / keep_prob
