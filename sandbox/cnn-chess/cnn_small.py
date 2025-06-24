# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


class ChessCNN(nn.Module):
    def __init__(self,
                 conv_layers=4,  # 3-6 layers recommended
                 init_filters=64,  # 32-128
                 dense_units=256,  # 128-512
                 dropout=0.3,  # 0.2-0.5
                 activation='lrelu'):  # lrelu/swish
        super().__init__()

        layers = []
        in_channels = 18  # 8x8x18 input (6 pieces * 3 colors + 5 meta + 1 turn)

        # Convolutional block
        for i in range(conv_layers):
            layers += [
                nn.Conv2d(in_channels, init_filters * (2 ** i), 3, padding=1),
                nn.BatchNorm2d(init_filters * (2 ** i)),
                nn.LeakyReLU(0.1) if activation == 'lrelu' else nn.SiLU(),
                nn.Dropout2d(dropout / 2)
            ]
            in_channels = init_filters * (2 ** i)
            if i % 2 == 1:
                layers.append(nn.MaxPool2d(2))  # Strategic downsampling

        self.conv = nn.Sequential(*layers)

        # Policy head (move prediction)
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels, 2, 1),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4672)  # All possible chess moves
        )

        # Value head (position evaluation)
        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Flatten(),
            nn.Linear(8 * 8, dense_units),
            nn.Linear(dense_units, 1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.conv(x)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value
