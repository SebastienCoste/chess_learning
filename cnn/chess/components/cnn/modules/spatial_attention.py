import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn.chess.components.config import TRAINING_CONFIG


class SimplifiedSelfAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super(SimplifiedSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Reduced complexity with fewer heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Stronger normalization layers around attention
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Apply normalization before attention (norm-first approach)
        x_norm = self.norm1(x)

        # Project to queries, keys, values
        qkv = self.qkv_proj(x_norm)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)

        # Apply second normalization
        out = self.norm2(out + x)

        return out


class SpatialChannelAttention(nn.Module):
    """
    Spatial attention mechanism for chess board regions.
    Generates attention weights for each spatial location.
    """

    def __init__(self, in_channels, reduction_ratio=8):
        super(SpatialChannelAttention, self).__init__()

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


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=TRAINING_CONFIG["spatial_kernel_size"]):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.regulation = nn.GELU() #nn.Sigmoid()

    def forward(self, x):
        # Generate spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)

        # Apply spatial attention
        return x * self.regulation(y)