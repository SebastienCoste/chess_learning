import torch.nn.functional as F
import torch.nn as nn
import torch

from cnn.chess.components.config import TRAINING_CONFIG
from cnn.chess.components.cnn.data_manip.mish_activation import MishActivation
from cnn.chess.components.cnn.modules.residual_block import ResidualBlock
from cnn.chess.components.cnn.modules.spatial_attention import SpatialChannelAttention
from cnn.chess.components.cnn.modules.chess_transformer_block import ChessTransformerBlock
from cnn.chess.components.cnn.modules.positional_encoding import PositionalEncoding2D


def get_activation_function(activation_name='gelu'):
    """Factory function to get activation functions."""
    activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'mish': MishActivation,
        'swish': lambda: nn.SiLU(),  # SiLU is equivalent to Swish
    }
    return activations.get(activation_name.lower(), nn.GELU)

class EnhancedChessCNN(nn.Module):
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
            use_attention=True,
            use_transformer_blocks=True,
            num_transformer_layers=2,
            transformer_heads=8,
            kernel_size = TRAINING_CONFIG["kernel_size"],
    ):
        super(EnhancedChessCNN, self).__init__()
        self.device = torch.device('cuda' if TRAINING_CONFIG["device"] == "cuda" and torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # Move entire model to CUDA immediately
        print(f"EnhancedChessCNN is initialized using device {self.device}")

        self.input_channels = input_channels
        self.board_size = board_size
        self.conv_filters = conv_filters
        self.fc_layers = fc_layers
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.use_attention = use_attention
        self.use_transformer_blocks = use_transformer_blocks
        self.kernel_size = kernel_size

        # Get activation function
        activation_fn = get_activation_function(activation)

        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(input_channels, board_size, board_size)
        # Convolutional layers with residual blocks
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels

        for filters in conv_filters:
            block = ResidualBlock(
                in_channels,
                filters,
                activation_fn=activation_fn,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate,
                kernel_size=self.kernel_size,
            )
            self.conv_layers.append(block)

            # Add attention after each residual block
            if use_attention:
                attention = SpatialChannelAttention(filters)
                self.conv_layers.append(attention)

            in_channels = filters

        # Calculate flattened size
        self.flattened_size = in_channels * board_size * board_size

        # Transformer blocks (optional)
        if use_transformer_blocks:
            self.transformer_blocks = nn.ModuleList([
                ChessTransformerBlock(
                    embed_dim=in_channels,
                    num_heads=transformer_heads,
                    dropout=dropout_rate
                ) for _ in range(num_transformer_layers)
            ])

        # Fully connected layers
        self.fc_layers_list = nn.ModuleList()
        in_features = self.flattened_size

        for fc_size in fc_layers:
            self.fc_layers_list.append(nn.Linear(in_features, fc_size))
            self.fc_layers_list.append(activation_fn())
            self.fc_layers_list.append(nn.Dropout(dropout_rate))
            in_features = fc_size

        # Output layer
        self.output_layer = nn.Linear(in_features, 64 * 64)

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
        # Add positional encoding
        x = self.pos_encoding(x)

        # Convolutional layers with residual connections and attention
        for layer in self.conv_layers:
            x = layer(x)

        # Transformer blocks (if enabled)
        if self.use_transformer_blocks:
            batch_size, channels, height, width = x.size()
            # Reshape for transformer: (batch, seq_len, embed_dim)
            x_reshaped = x.view(batch_size, channels, -1).transpose(1, 2)

            for transformer in self.transformer_blocks:
                x_reshaped = transformer(x_reshaped)

            # Reshape back to conv format
            x = x_reshaped.transpose(1, 2).view(batch_size, channels, height, width)

        # Flatten
        #can't run x = x.view(x.size(0), -1) because of:
        # view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        # so might also write x = x.contiguous().view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        for layer in self.fc_layers_list:
            x = layer(x)

        # Output layer
        x = self.output_layer(x)

        return x

    def get_move_probabilities(self, x):
        """Get move probabilities with temperature scaling."""
        logits = self.forward(x)
        logits = logits.view(logits.size(0), 64, 64)
        probabilities = F.softmax(logits.view(logits.size(0), -1), dim=1)
        return probabilities.view(logits.size(0), 64, 64)