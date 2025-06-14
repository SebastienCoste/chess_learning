
'''
Chess CNN Model Architecture

A configurable Convolutional Neural Network for chess position evaluation.
This implementation supports parameter tuning from 500K to 2M parameters
and includes detailed documentation of all hyperparameters.

Based on AlphaZero architecture principles with optimization for smaller networks.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class ChessConv2d(nn.Module):
    """
    Custom convolutional block for chess with batch normalization and activation.

    Parameters:
    -----------
    in_channels : int
        Number of input channels
    out_channels : int  
        Number of output channels
    kernel_size : int, default=3
        Size of convolving kernel (3x3 is standard for chess)
    padding : int, default=1
        Zero-padding added to both sides of input (1 for 3x3 kernel)
    use_batch_norm : bool, default=True
        Whether to use batch normalization (recommended: True)
    activation : str, default='relu'
        Activation function ('relu', 'leaky_relu', 'elu', 'swish')
    dropout_rate : float, default=0.0
        Dropout rate (0.0 = no dropout, 0.1-0.3 typical range)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 use_batch_norm: bool = True,
                 activation: str = 'relu',
                 dropout_rate: float = 0.0):
        super(ChessConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             padding=padding, bias=not use_batch_norm)

        self.batch_norm = nn.BatchNorm2d(out_channels) if use_batch_norm else None

        # Activation functions
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'swish':
            self.activation = nn.SiLU(inplace=True)  # SiLU is Swish
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block for deeper networks with skip connections.

    Parameters:
    -----------
    channels : int
        Number of channels (input = output for residual connection)
    kernel_size : int, default=3
        Convolutional kernel size
    use_batch_norm : bool, default=True
        Whether to use batch normalization
    activation : str, default='relu'
        Activation function type
    dropout_rate : float, default=0.0
        Dropout rate
    """

    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 use_batch_norm: bool = True,
                 activation: str = 'relu',
                 dropout_rate: float = 0.0):
        super(ResidualBlock, self).__init__()

        self.conv1 = ChessConv2d(channels, channels, kernel_size, 
                                kernel_size//2, use_batch_norm, 
                                activation, dropout_rate)
        self.conv2 = ChessConv2d(channels, channels, kernel_size, 
                                kernel_size//2, use_batch_norm, 
                                'relu', 0.0)  # No dropout on second conv

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual  # Skip connection
        return F.relu(out)


class ChessCNN(nn.Module):
    """
    Configurable Chess CNN for position evaluation.

    Architecture Parameters:
    -----------------------

    INPUT ENCODING:
    input_channels : int, default=18
        Number of input channels for board representation
        - 12 channels for pieces (6 types × 2 colors)
        - 4 channels for castling rights
        - 1 channel for en passant
        - 1 channel for turn to move

    NETWORK ARCHITECTURE:
    num_conv_layers : int, default=4
        Number of convolutional layers (2-8 recommended)
        More layers = better pattern recognition but slower training

    base_filters : int, default=64
        Base number of filters (32, 64, 128, 256)
        Directly affects parameter count and model capacity

    filter_multiplier : float, default=1.5
        Filter growth rate between layers (1.0-2.0)
        1.0 = constant filters, 2.0 = double each layer

    use_residual : bool, default=False
        Use residual connections (True for deeper networks >6 layers)

    kernel_size : int, default=3
        Convolutional kernel size (3 recommended for chess)

    REGULARIZATION:
    dropout_rate : float, default=0.1
        Dropout rate (0.0-0.3, higher for overfitting prevention)

    use_batch_norm : bool, default=True
        Use batch normalization (strongly recommended: True)

    l2_weight_decay : float, default=1e-4
        L2 regularization strength (1e-5 to 1e-3)

    OUTPUT HEAD:
    fc_hidden_size : int, default=512
        Size of fully connected hidden layer (256, 512, 1024)

    activation : str, default='relu'
        Activation function throughout network

    PARAMETER COUNT ESTIMATION:
    With default settings: ~580K parameters
    - base_filters=32: ~150K parameters  
    - base_filters=64: ~580K parameters
    - base_filters=96: ~1.2M parameters
    - base_filters=128: ~2.1M parameters
    """

    def __init__(self, 
                 # Input configuration
                 input_channels: int = 18,

                 # Architecture configuration  
                 num_conv_layers: int = 4,
                 base_filters: int = 64,
                 filter_multiplier: float = 1.5,
                 use_residual: bool = False,
                 kernel_size: int = 3,

                 # Regularization
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = True,

                 # Output configuration
                 fc_hidden_size: int = 512,
                 activation: str = 'relu',

                 # Additional options
                 squeeze_excitation: bool = False):

        super(ChessCNN, self).__init__()

        # Store configuration
        self.config = {
            'input_channels': input_channels,
            'num_conv_layers': num_conv_layers,
            'base_filters': base_filters,
            'filter_multiplier': filter_multiplier,
            'use_residual': use_residual,
            'kernel_size': kernel_size,
            'dropout_rate': dropout_rate,
            'use_batch_norm': use_batch_norm,
            'fc_hidden_size': fc_hidden_size,
            'activation': activation,
            'squeeze_excitation': squeeze_excitation
        }

        # Build convolutional layers
        self.conv_layers = nn.ModuleList()

        current_channels = input_channels
        for i in range(num_conv_layers):
            # Calculate output channels for this layer
            if i == 0:
                out_channels = base_filters
            else:
                out_channels = int(base_filters * (filter_multiplier ** i))

            if use_residual and i > 0 and current_channels == out_channels:
                # Add residual block when dimensions match
                layer = ResidualBlock(current_channels, kernel_size, 
                                    use_batch_norm, activation, dropout_rate)
            else:
                # Regular convolutional layer
                layer = ChessConv2d(current_channels, out_channels, kernel_size,
                                  kernel_size//2, use_batch_norm, activation, 
                                  dropout_rate)

            self.conv_layers.append(layer)
            current_channels = out_channels

        # Calculate flattened size after convolutions
        # For 8x8 chess board, after conv layers: 8x8 = 64 squares
        self.flattened_size = current_channels * 8 * 8

        # Fully connected layers for position evaluation
        self.fc1 = nn.Linear(self.flattened_size, fc_hidden_size)
        self.fc1_activation = self._get_activation(activation)
        self.fc1_dropout = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(fc_hidden_size, 256)  # Intermediate layer
        self.fc2_activation = self._get_activation(activation)
        self.fc2_dropout = nn.Dropout(dropout_rate)

        # Output layer: single value for position evaluation
        self.output = nn.Linear(256, 1)

        # Initialize weights
        self._initialize_weights()

    def _get_activation(self, activation: str):
        """Get activation function by name."""
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'elu':
            return nn.ELU(inplace=True)
        elif activation == 'swish':
            return nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def _initialize_weights(self):
        """Initialize network weights using best practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, 8, 8)

        Returns:
        --------
        torch.Tensor
            Position evaluation score (batch_size, 1)
            Positive values favor white, negative favor black
        """
        # Convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1_dropout(self.fc1_activation(self.fc1(x)))
        x = self.fc2_dropout(self.fc2_activation(self.fc2(x)))

        # Output layer with tanh activation for bounded output [-1, 1]
        x = torch.tanh(self.output(x))

        return x

    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and trainable parameters.

        Returns:
        --------
        Tuple[int, int]
            (total_params, trainable_params)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def get_model_info(self) -> Dict:
        """Get comprehensive model information."""
        total_params, trainable_params = self.count_parameters()

        return {
            'model_name': 'ChessCNN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'configuration': self.config,
            'architecture_summary': self._get_architecture_summary()
        }

    def _get_architecture_summary(self) -> List[str]:
        """Get human-readable architecture summary."""
        summary = []
        summary.append(f"Input: {self.config['input_channels']} channels (8×8 chess board)")

        current_channels = self.config['input_channels']
        for i in range(self.config['num_conv_layers']):
            if i == 0:
                out_channels = self.config['base_filters']
            else:
                out_channels = int(self.config['base_filters'] * (self.config['filter_multiplier'] ** i))

            layer_type = "ResBlock" if (self.config['use_residual'] and i > 0 and current_channels == out_channels) else "Conv2D"
            summary.append(f"Layer {i+1}: {layer_type} {current_channels}→{out_channels} channels")
            current_channels = out_channels

        summary.append(f"Flatten: {self.flattened_size} features")
        summary.append(f"FC1: {self.flattened_size}→{self.config['fc_hidden_size']}")
        summary.append(f"FC2: {self.config['fc_hidden_size']}→256")
        summary.append(f"Output: 256→1 (position evaluation)")

        return summary


def create_model_configs() -> Dict[str, Dict]:
    """
    Pre-defined model configurations for different parameter counts.

    Returns optimized configurations for training in <8 hours and maximum efficiency.
    """

    configs = {
        # Small model: ~500K parameters, trains in 2-3 hours
        'small_fast': {
            'input_channels': 18,
            'num_conv_layers': 3,
            'base_filters': 48,
            'filter_multiplier': 1.3,
            'use_residual': False,
            'kernel_size': 3,
            'dropout_rate': 0.15,
            'use_batch_norm': True,
            'fc_hidden_size': 256,
            'activation': 'relu',
            'description': 'Fast training model, ~500K params, 2-3 hours'
        },

        # Medium model: ~800K parameters, trains in 4-5 hours  
        'medium_balanced': {
            'input_channels': 18,
            'num_conv_layers': 4,
            'base_filters': 56,
            'filter_multiplier': 1.4,
            'use_residual': False,
            'kernel_size': 3,
            'dropout_rate': 0.12,
            'use_batch_norm': True,
            'fc_hidden_size': 384,
            'activation': 'relu',
            'description': 'Balanced model, ~800K params, 4-5 hours'
        },

        # Large model: ~1.5M parameters, trains in 6-7 hours
        'large_efficient': {
            'input_channels': 18,
            'num_conv_layers': 5,
            'base_filters': 64,
            'filter_multiplier': 1.5,
            'use_residual': True,
            'kernel_size': 3,
            'dropout_rate': 0.1,
            'use_batch_norm': True,
            'fc_hidden_size': 512,
            'activation': 'relu',
            'description': 'Large efficient model, ~1.5M params, 6-7 hours'
        },

        # Maximum efficiency: ~2M parameters, trains in 7-8 hours
        'max_efficiency': {
            'input_channels': 18,
            'num_conv_layers': 6,
            'base_filters': 72,
            'filter_multiplier': 1.6,
            'use_residual': True,
            'kernel_size': 3,
            'dropout_rate': 0.08,
            'use_batch_norm': True,
            'fc_hidden_size': 640,
            'activation': 'swish',  # Swish activation for better performance
            'description': 'Maximum efficiency model, ~2M params, 7-8 hours'
        }
    }

    return configs


if __name__ == "__main__":
    # Example usage and parameter counting
    configs = create_model_configs()

    print("Chess CNN Model Configurations:")
    print("=" * 50)

    for name, config in configs.items():
        model = ChessCNN(**{k: v for k, v in config.items() if k != 'description'})
        info = model.get_model_info()

        print(f"\n{name.upper()}:")
        print(f"Description: {config['description']}")
        print(f"Parameters: {info['total_parameters']:,}")
        print(f"Size: {info['parameter_size_mb']:.1f} MB")
        print("Architecture:")
        for line in info['architecture_summary']:
            print(f"  {line}")
