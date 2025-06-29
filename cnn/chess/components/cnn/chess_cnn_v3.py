import torch.nn.functional as F
import torch.nn as nn
import torch

from cnn.chess.components.config import TRAINING_CONFIG
from cnn.chess.components.cnn.modules.dense_block import DenseBlock, TransitionLayer
from cnn.chess.components.cnn.modules.mish_activation import MishActivation
from cnn.chess.components.cnn.modules.multi_scale_feature_extraction import MultiScaleConv
from cnn.chess.components.cnn.modules.residual_block import SEResidualBlock
from cnn.chess.components.cnn.modules.spatial_attention import SpatialAttention, SimplifiedSelfAttention
from cnn.chess.components.cnn.modules.positional_encoding import PositionalEncoding2D
from cnn.chess.components.cnn.modules.stochastic_depth import StochasticDepth


def get_activation_function(activation_name='gelu'):
    """Factory function to get activation functions."""
    activations = {
        'relu': lambda: nn.ReLU(inplace=True),
        'gelu': lambda: nn.GELU(),
        'mish': lambda: MishActivation(inplace=True),
        'swish': lambda: nn.SiLU(),  # SiLU is equivalent to Swish
    }
    return activations.get(activation_name.lower(), lambda: nn.GELU())

class EnhancedChessCNNV3(nn.Module):
    """
    Enhanced Convolutional Neural Network for chess move prediction.
    Incorporates residual connections, modern activations, attention mechanisms,
    and transformer-inspired components with comprehensive W&B integration.
    """

    def __init__(
            self,
            input_channels=TRAINING_CONFIG["input_channels"],
            board_size=TRAINING_CONFIG["board_size"],
            conv_filters=[64, 128, 256],
            fc_layers=[512, 256],
            dropout_rate=0.3,
            batch_norm=True,
            activation='gelu',
            transformer_heads=8,
            kernel_size = TRAINING_CONFIG["kernel_size"],
    ):
        super(EnhancedChessCNNV3    , self).__init__()
        self.device = torch.device('cuda' if TRAINING_CONFIG["device"] == "cuda" and torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # Move entire model to CUDA immediately
        print(f"EnhancedChessCNN is initialized using device {self.device}")

        self.input_channels = input_channels
        self.board_size = board_size
        self.conv_filters = conv_filters
        self.fc_layers = fc_layers
        self.batch_norm = batch_norm
        self.kernel_size = kernel_size

        # Get activation function
        activation_fn = get_activation_function(activation)

        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(input_channels, board_size, board_size)
        # Initial convolution
        self.initial_conv = MultiScaleConv(input_channels, 64, activation_fn)
        self.spatial_dropout1 = nn.Dropout2d(p=0.3)  # Spatial dropout for conv layers
        # Dense blocks with transition layers
        self.dense_block1 = DenseBlock(64, growth_rate=32, num_layers=4, activation_fn = activation_fn)
        in_channels = 64 + 4 * 32  # Initial + growth_rate * num_layers
        self.transition1 = TransitionLayer(in_channels, in_channels // 2, activation_fn = activation_fn)

        # SE-ResNet block
        self.se_block = SEResidualBlock(in_channels // 2, activation_fn = activation_fn)

        # Spatial attention
        self.spatial_attn = SpatialAttention()

        # Simplified self-attention
        self.self_attn = SimplifiedSelfAttention(embed_dim=96, num_heads=transformer_heads) #or embed_dim = 128 ?

        # Stochastic depth
        self.stochastic_depth = StochasticDepth(drop_prob=0.1)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Output layers
        self.flatten = nn.Flatten()
        #below is incorrect because of RuntimeError: mat1 and mat2 shapes cannot be multiplied (1024x1536 and 6144x4096)
        # at x = self.fc(x)
        #self.fc = nn.Linear((in_channels // 2) * board_size * board_size, 64 * 64)
        # because nn.AvgPool2d(kernel_size=2, stride=2)  reduces the board_size from 8 to 4
        #Below is the correct one but fixed
        # self.fc = nn.Linear((in_channels // 2) * (board_size // 2) * (board_size // 2), 64 * 64)
        # Below is the dynamic one
        with torch.no_grad():
            dummy_input = torch.randn(1, TRAINING_CONFIG["input_channels"], TRAINING_CONFIG["board_size"], TRAINING_CONFIG["board_size"])
            x = dummy_input
            #Reproducing what we have above
            x = self.pos_encoding(x)
            x = self.initial_conv(x)
            x = self.dense_block1(x)
            x = self.transition1(x)
            identity = x
            x = self.se_block(x)
            x = self.stochastic_depth(x)
            x = x + identity
            x = self.spatial_attn(x)
            batch, channels, height, width = x.shape
            features = channels * height * width
            self.fc = nn.Linear(features, 64 * 64)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Multi-scale feature extraction
        # Apply positional encoding FIRST
        x = self.pos_encoding(x)

        x = self.initial_conv(x)
        x = self.spatial_dropout1(x)  # Apply after activation

        # Dense connectivity
        x = self.dense_block1(x)
        x = self.dropout1(x)  # Regular dropout for FC layers
        x = self.transition1(x)

        # SE-ResNet with stochastic depth
        identity = x
        x = self.se_block(x)
        x = self.stochastic_depth(x)
        x = x + identity

        # Spatial attention
        x = self.spatial_attn(x)

        # Reshape for self-attention
        batch, channels, height, width = x.shape
        x = x.view(batch, channels, -1).transpose(1, 2)

        # Simplified self-attention
        x = self.self_attn(x)

        # Reshape back and output
        x = x.transpose(1, 2).view(batch, channels, height, width)
        x = self.flatten(x)
        #print(f"Flattened shape: {x.shape}")  # Should be [batch, 1536]
        x = self.fc(x)

        return x

    def get_move_probabilities(self, x):
        """Get move probabilities with temperature scaling."""
        logits = self.forward(x)
        logits = logits.view(logits.size(0), 64, 64)
        probabilities = F.softmax(logits.view(logits.size(0), -1), dim=1)
        return probabilities.view(logits.size(0), 64, 64)