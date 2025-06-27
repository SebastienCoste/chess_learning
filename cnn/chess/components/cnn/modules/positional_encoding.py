import torch
import torch.nn as nn
import math

from cnn.chess.components.config import TRAINING_CONFIG


class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding for chess board positions.
    Encodes spatial relationships between squares.
    """

    def __init__(self, channels, height=8, width=8):
        super(PositionalEncoding2D, self).__init__()
        pe = torch.zeros(channels, height, width)

        # Create position encodings
        for i in range(height):
            for j in range(width):
                for k in range(0, channels, 4):
                    if k < channels:
                        pe[k, i, j] = math.sin(i / (10000 ** (k / channels)))
                    if k + 1 < channels:
                        pe[k + 1, i, j] = math.cos(i / (10000 ** (k / channels)))
                    if k + 2 < channels:
                        pe[k + 2, i, j] = math.sin(j / (10000 ** (k / channels)))
                    if k + 3 < channels:
                        pe[k + 3, i, j] = math.cos(j / (10000 ** (k / channels)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :x.size(2), :x.size(3)]