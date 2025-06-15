import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for chess board regions.
    Generates attention weights for each spatial location.
    """

    def __init__(self, in_channels, reduction_ratio=8):
        super(SpatialAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Channel attention
        avg_pool = self.avg_pool(x).view(batch_size, channels)
        max_pool = self.max_pool(x).view(batch_size, channels)
        channel_input = torch.cat([avg_pool, max_pool], dim=1)
        channel_weights = self.channel_attention(channel_input).view(batch_size, channels, 1, 1)

        # Apply channel attention
        x = x * channel_weights

        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_weights = self.spatial_conv(spatial_input)

        # Apply spatial attention
        x = x * spatial_weights

        return x