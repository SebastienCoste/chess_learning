import torch
import torch.nn as nn

"""
Multi-scale feature extraction captures patterns at different scales simultaneously . 
For chess, this helps recognize both local tactical patterns (3×3 kernels) and broader strategic relationships (5×5 kernels).
"""
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        out = torch.cat([out3x3, out5x5], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out
