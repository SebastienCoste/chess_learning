import torch                     # Main PyTorch library for tensor operations
import torch.nn as nn            # Neural network modules (layers, loss functions)
import torch.nn.functional as F  # Functional interface for operations like activation functions
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import math

BASE_CONFIG = {
    "input_channels" : 19, #12 pieces, 1 en passant, 4 castling, check, light/dark square bishop
    "board_size" : 8,
    "kernel_size" : 3,
    "stride" : 1,
    'activation_fn' : "gelu",
}
# Example model configurations for different time constraints
"""
The model's parameters are distributed across different layers, with the fully connected layers containing the vast majority of parameters:
    Conv Layer 1: ~8,256 parameters (0.2% of total)
    Conv Layer 2: ~41,568 parameters (1.0% of total)
    FC Layer: ~2,359,680 parameters (59.2% of total)
    Output Layer: ~1,576,960 parameters (39.6% of total)
Total: ~3,986,464 parameters

This distribution is typical for CNN architectures, where fully connected layers often contain most of the parameters.
"""
FAST_TRAINING_CONFIG = {
    **BASE_CONFIG,                # Inherits all base configuration parameters
    'conv_filters': [48, 96],     # Two convolutional layers with 48 and 96 filters
    'fc_layers': [384],           # One fully connected layer with 384 neurons
    'dropout_rate': 0.2,          # 20% dropout for regularization
    'batch_norm': False,          # No batch normalization
    'use_attention': False,
    'use_transformer_blocks': False,
}

OPTIMAL_TRAINING_CONFIG = {
    **BASE_CONFIG,
    'conv_filters': [64, 128, 256], # Three convolutional layers
    'fc_layers': [512, 256],        # Two fully connected layers
    'dropout_rate': 0.3,            # 30% dropout
    'batch_norm': True,             # With batch normalization
    'use_attention': True,
    'use_transformer_blocks': True,
    'num_transformer_layers': 2,
    'transformer_heads': 8,
}

CONFIG = FAST_TRAINING_CONFIG      # Sets the active configuration


class ResidualBlock(nn.Module):
    """
    Residual block with skip connections for improved gradient flow.
    Implements the identity mapping approach from ResNet architecture.
    """
    def __init__(self,
                 out_channels,
                 in_channels = CONFIG["input_channels"],
                 kernel_size= CONFIG["kernel_size"],
                 stride= CONFIG["stride"],
                 activation_fn= nn.GELU if CONFIG["activation_fn"]=="gelu" else nn.ReLU,
                 batch_norm=CONFIG["batch_norm"],
                 dropout_rate=CONFIG["dropout_rate"],
                 ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,
                               padding=1,
                               bias=not batch_norm)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size,
                               1,
                               padding=1,
                               bias=not batch_norm)

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

class ChessCNN(nn.Module):
    """
    Convolutional Neural Network for chess move prediction.
    
    This model takes a chess board representation as input and 
    outputs a move probability distribution.
    
    Parameters:
    -----------
    input_channels : int
        Number of input channels (typically 12 for 6 piece types x 2 colors)
    board_size : int
        Size of the chess board (typically 8 for standard chess)
    conv_filters : list
        List of integers representing the number of filters in each convolutional layer
    fc_layers : list
        List of integers representing the size of each fully connected layer
    dropout_rate : float
        Dropout rate for regularization
    batch_norm : bool
        Whether to use batch normalization
    """
    def __init__(
        self, 
        input_channels=CONFIG["input_channels"],
        out_channels=64 * 64,
        board_size=CONFIG["board_size"],
        conv_filters=CONFIG["conv_filters"], #[64, 128, 256],
        fc_layers=CONFIG["fc_layers"], #[1024, 512],
        dropout_rate=CONFIG["dropout_rate"], #0.3,
        batch_norm=CONFIG["batch_norm"],
        kernel_size=CONFIG["kernel_size"],
        stride= CONFIG["stride"],
        activation_fn= nn.GELU if CONFIG["activation_fn"]=="gelu" else nn.ReLU,
        use_attention=CONFIG["use_attention"],
        use_transformer_blocks=CONFIG["use_attention"],
        num_transformer_layers=2,
        transformer_heads=8
    ):
        super(ChessCNN, self).__init__()

        self.input_channels = input_channels
        self.out_channels = out_channels
        self.board_size = board_size
        self.conv_filters = conv_filters
        self.fc_layers = fc_layers
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation_fn()
        self.use_attention = use_attention
        self.use_transformer_blocks = use_transformer_blocks

        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(input_channels, board_size, board_size)

        # Calculate parameter count to ensure it falls within 500K-2M range
        self._create_layers()
        self.param_count = self._count_parameters()
        
    def _create_layers(self):
        """Create all layers of the CNN model.
        This loop creates the convolutional layers based on the conv_filters configuration:
            For each filter size in conv_filters, it creates a convolutional layer
            If batch normalization is enabled, it adds a BatchNorm2d layer
            It adds a ReLU activation function for non-linearity
            It adds a Dropout layer for regularization
            It updates in_channels for the next layer
        The method then calculates the size of the flattened features and creates fully connected layers:"""
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = self.input_channels
        # Skip connection adjustment for dimension matching
        self.skip_connection = None
        if in_channels != self.out_channels or self.stride != 1:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels, 1, self.stride, bias=False),
                nn.BatchNorm2d(self.out_channels) if self.batch_norm else nn.Identity()
            )

        for i, filters in enumerate(self.conv_filters):
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=self.kernel_size,
                padding=1
            )
            self.conv_layers.append(conv_layer)
            
            if self.batch_norm:
                self.conv_layers.append(nn.BatchNorm2d(filters))
            
            self.conv_layers.append(self.activation())
            self.conv_layers.append(nn.Dropout2d(self.dropout_rate))
            
            in_channels = filters
        
        # Calculate the size of the flattened features after convolution
        """The method then calculates the size of the flattened features and creates fully connected layers
        This loop creates the fully connected layers based on the fc_layers configuration:
            For each size in fc_layers, it creates a linear layer
            It adds a ReLU activation function
            It adds a Dropout layer for regularization
            It updates in_features for the next layer
        """
        self.flattened_size = in_channels * self.board_size * self.board_size
        
        # Fully connected layers
        self.fc_layers_list = nn.ModuleList()
        in_features = self.flattened_size
        
        for i, fc_size in enumerate(self.fc_layers):
            fc_layer = nn.Linear(in_features, fc_size)
            self.fc_layers_list.append(fc_layer)
            self.fc_layers_list.append(nn.ReLU())
            self.fc_layers_list.append(nn.Dropout(self.dropout_rate))
            in_features = fc_size


        # Output layer - for move prediction (from-square and to-square)
        # 64 squares for from-square, 64 squares for to-square
        """
        Finally, it creates the output layer
        This output layer maps to 4096 values, representing all possible moves from any square to any square on the chess board (64×64)
        """
        self.output_layer = nn.Linear(in_features, self.out_channels)
        
    def forward(self, x):
        """
        Forward pass through the network.
        The forward method defines how data flows through the network
        This method processes the input tensor through all layers sequentially:
            It passes the input through all convolutional layers
            It flattens the output to a 1D vector
            It passes the flattened output through all fully connected layers
            It passes through the output layer to produce the final output

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor representing a batch of chess boards
            Shape: (batch_size, input_channels, board_size, board_size)
            
        Returns:
        --------
        torch.Tensor
            Output tensor with move probabilities
            Shape: (batch_size, 64*64) for standard chess
        """
        # Convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for layer in self.fc_layers_list:
            x = layer(x)

        if self.skip_connection is not None:
            x = self.skip_connection(x)

        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def _count_parameters(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_move_probabilities(self, x):
        """
        Get move probabilities from model output.
        converts the raw model output to move probabilities:
            Gets the raw logits from the forward pass
            Reshapes them to (batch_size, 64, 64) for from-square and to-square representation
            Applies softmax to get a probability distribution
            Returns the probabilities in the same shape

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor representing a batch of chess boards
            
        Returns:
        --------
        torch.Tensor
            Probability distribution over all possible moves
        """
        logits = self.forward(x)
        # Reshape to (batch_size, 64, 64) for from-square and to-square
        logits = logits.view(logits.size(0), 64, 64)
        probabilities = F.softmax(logits.view(logits.size(0), -1), dim=1)
        return probabilities.view(logits.size(0), 64, 64)
    
    def adjust_for_parameter_target(self, target_param_range=(500000, 2000000)):
        """
        Adjust model architecture to fit within target parameter range.
        This method ensures the model has an appropriate number of parameters:
            If the model has too few parameters, it scales up the layer sizes
            If the model has too many parameters, it scales down the layer sizes
            It returns the adjusted model and its parameter count

        Parameters:
        -----------
        target_param_range : tuple
            Target range for parameter count (min, max)
            
        Returns:
        --------
        tuple
            (adjusted_model, param_count)
        """
        min_params, max_params = target_param_range
        
        if self.param_count < min_params:
            # Increase model size
            scale_factor = (min_params / self.param_count) ** 0.5
            new_conv_filters = [max(64, int(f * scale_factor)) for f in self.conv_filters]
            new_fc_layers = [max(128, int(f * scale_factor)) for f in self.fc_layers]
            
            adjusted_model = ChessCNN(
                input_channels=self.input_channels,
                board_size=self.board_size,
                conv_filters=new_conv_filters,
                fc_layers=new_fc_layers,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm
            )
            
        elif self.param_count > max_params:
            # Decrease model size
            scale_factor = (max_params / self.param_count) ** 0.5
            new_conv_filters = [max(32, int(f * scale_factor)) for f in self.conv_filters]
            new_fc_layers = [max(64, int(f * scale_factor)) for f in self.fc_layers]
            
            adjusted_model = ChessCNN(
                input_channels=self.input_channels,
                board_size=self.board_size,
                conv_filters=new_conv_filters,
                fc_layers=new_fc_layers,
                dropout_rate=self.dropout_rate,
                batch_norm=self.batch_norm
            )
        else:
            # Model already in target range
            adjusted_model = self
            
        return adjusted_model, adjusted_model.param_count


def create_small_cnn_model(param_target=(500000, 2000000)):
    """
    Create a small CNN model with parameter count in the target range.
    
    Parameters:
    -----------
    param_target : tuple
        Target range for parameter count (min, max)
        
    Returns:
    --------
    ChessCNN
        Model with parameter count in target range
    """
    # Start with a small model
    model = ChessCNN(
        #input_channels=12,  # 6 piece types × 2 colors
        #board_size=8,
        #conv_filters=[64, 128],
        #fc_layers=[512],
        #dropout_rate=0.3,
        #batch_norm=True
    )
    
    # Adjust to target parameter range
    adjusted_model, param_count = model.adjust_for_parameter_target(param_target)

    print(f"Created CNN model with {param_count:,} parameters")
    
    return adjusted_model


if __name__ == "__main__":
    # Example usage and parameter counting
    print(torch.__version__)
    create_small_cnn_model()