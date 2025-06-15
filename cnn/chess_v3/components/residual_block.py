import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Residual block with skip connections for improved gradient flow.
    Implements the identity mapping approach from ResNet architecture.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 activation_fn=nn.GELU, batch_norm=True, dropout_rate=0.2):
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