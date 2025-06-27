import torch
import torch.nn as nn

"""
Dense connections allow each layer to receive feature maps from all preceding layers . This encourages feature reuse, strengthens gradient flow, and increases effective depth without adding many parameters.
"""
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, activation_fn):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                activation_fn(), #nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1)
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation_fn):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            activation_fn(), #nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)
