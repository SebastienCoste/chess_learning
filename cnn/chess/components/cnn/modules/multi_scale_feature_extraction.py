import torch
import torch.nn as nn

"""
Multi-scale feature extraction captures patterns at different scales simultaneously . 
For chess, this helps recognize both local tactical patterns (3×3 kernels) and broader strategic relationships (5×5 kernels).
"""
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation_fn):
        super(MultiScaleConv, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation_fn() # nn.ReLU(inplace=True)

    def forward(self, x):
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        out = torch.cat([out3x3, out5x5], dim=1)
        out = self.bn(out)
        out = self.activation(out)
        return out


class PostAttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation_fn):
        super().__init__()
        # Depthwise separable convolution for efficiency
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Multi-scale residual connections
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

        # Gated mechanism
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 3, out_channels // 4, kernel_size=1),
            activation_fn(),
            nn.Conv2d(out_channels // 4, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation_fn()

    def forward(self, x):
        # Depthwise separable base
        base = self.pointwise(self.depthwise(x))

        # Multi-scale features
        scale1 = self.conv1x1(x)
        scale3 = self.conv3x3(x)
        scale5 = self.conv5x5(x)

        # Gated fusion
        weights = self.gate(torch.cat([scale1, scale3, scale5], dim=1))
        fused = weights[:, 0:1] * scale1 + weights[:, 1:2] * scale3 + weights[:, 2:3] * scale5

        # Residual connection
        out = base + fused
        out = self.bn(out)
        return self.activation(out)
