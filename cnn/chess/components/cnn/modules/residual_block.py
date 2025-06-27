import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Residual block with skip connections for improved gradient flow.
    Implements the identity mapping approach from ResNet architecture.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 activation_fn=lambda: nn.GELU(), batch_norm=True, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride, padding=1, bias=not batch_norm)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               1, padding=1, bias=not batch_norm)

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

        self.activation = activation_fn()
        self.dropout = nn.Dropout2d(dropout_rate)

        # Skip connection adjustment for dimension matching
        self.skip_connection = None
        if in_channels != out_channels or stride != 1:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)

        # Apply skip connection transformation if needed
        if self.skip_connection is not None:
            residual = self.skip_connection(x)

        # Add residual connection
        out += residual
        out = self.activation(out)

        return out

"""
Squeeze-and-Excitation (SE) blocks enhance residual connections by recalibrating channel-wise feature responses. 
This improves gradient flow and feature representation by explicitly modeling interdependencies between channels.
"""
class SEResidualBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16, activation_fn=lambda: nn.ReLU(inplace=True)):
        super(SEResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = activation_fn()

        # Squeeze-and-Excitation block
        self.se = SqueezeExcitation(channels, reduction_ratio, activation_fn)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE block
        out = self.se(out)

        out += residual
        out = self.activation(out)

        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction_ratio=16, activation_fn=lambda: nn.ReLU(inplace=True)):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            activation_fn(),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1, 1)
        return x * y
