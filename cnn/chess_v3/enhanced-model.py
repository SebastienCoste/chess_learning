import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import math
import wandb
from torchinfo import summary
import warnings
from typing import Dict, List, Tuple, Optional, Any
import os
from dotenv import load_dotenv

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

INPUT_CHANNELS = 20 #0-6 white pieces with light/dark bishop distinction, 7-13 same for black, 14-17 castling right, 18 turn to move, 19 check state

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


class MishActivation(nn.Module):
    """
    Mish activation function implementation.
    Mish(x) = x * tanh(softplus(x))
    """

    def __init__(self):
        super().__init__()
        # Use built-in Mish if available (PyTorch 1.9+)
        if hasattr(F, 'mish'):
            self.act = F.mish
        else:
            self.act = self._mish_implementation

    def _mish_implementation(self, x):
        return x * torch.tanh(F.softplus(x))

    def forward(self, x):
        return self.act(x)


def get_activation_function(activation_name='gelu'):
    """Factory function to get activation functions."""
    activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'mish': MishActivation,
        'swish': lambda: nn.SiLU(),  # SiLU is equivalent to Swish
    }
    return activations.get(activation_name.lower(), nn.GELU)


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for chess board regions.
    Generates attention weights for each spatial location.
    """

    def __init__(self, in_channels, reduction_ratio=8):
        super(SpatialAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Channel attention
        avg_pool = self.avg_pool(x).view(batch_size, channels)
        max_pool = self.max_pool(x).view(batch_size, channels)
        channel_input = torch.cat([avg_pool, max_pool], dim=1)
        channel_weights = self.channel_attention(channel_input).view(batch_size, channels, 1, 1)

        # Apply channel attention
        x = x * channel_weights

        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_weights = self.spatial_conv(spatial_input)

        # Apply spatial attention
        x = x * spatial_weights

        return x


class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding for chess board positions.
    Encodes spatial relationships between squares.
    """

    def __init__(self, channels, height=8, width=8):
        super(PositionalEncoding2D, self).__init__()

        pe = torch.zeros(channels, height, width)

        # Create position encodings
        for i in range(height):
            for j in range(width):
                for k in range(0, channels, 4):
                    if k < channels:
                        pe[k, i, j] = math.sin(i / (10000 ** (k / channels)))
                    if k + 1 < channels:
                        pe[k + 1, i, j] = math.cos(i / (10000 ** (k / channels)))
                    if k + 2 < channels:
                        pe[k + 2, i, j] = math.sin(j / (10000 ** (k / channels)))
                    if k + 3 < channels:
                        pe[k + 3, i, j] = math.cos(j / (10000 ** (k / channels)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :x.size(2), :x.size(3)]


class ChessTransformerBlock(nn.Module):
    """
    Transformer-inspired block for chess position understanding.
    Combines self-attention with feed-forward processing.
    """

    def __init__(self, embed_dim, num_heads=8, ff_dim=None, dropout=0.1):
        super(ChessTransformerBlock, self).__init__()

        if ff_dim is None:
            ff_dim = embed_dim * 4

        self.attention = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class EnhancedChessCNN(nn.Module):
    """
    Enhanced Convolutional Neural Network for chess move prediction.
    Incorporates residual connections, modern activations, attention mechanisms,
    and transformer-inspired components with comprehensive W&B integration.
    """

    def __init__(
            self,
            input_channels=INPUT_CHANNELS,
            board_size=8,
            conv_filters=[64, 128, 256],
            fc_layers=[512, 256],
            dropout_rate=0.3,
            batch_norm=True,
            activation='gelu',
            use_attention=True,
            use_transformer_blocks=True,
            num_transformer_layers=2,
            transformer_heads=8
    ):
        super(EnhancedChessCNN, self).__init__()

        self.input_channels = input_channels
        self.board_size = board_size
        self.conv_filters = conv_filters
        self.fc_layers = fc_layers
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.use_attention = use_attention
        self.use_transformer_blocks = use_transformer_blocks

        # Get activation function
        activation_fn = get_activation_function(activation)

        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(input_channels, board_size, board_size)

        # Convolutional layers with residual blocks
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels

        for filters in conv_filters:
            block = ResidualBlock(
                in_channels, filters,
                activation_fn=activation_fn,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate
            )
            self.conv_layers.append(block)

            # Add attention after each residual block
            if use_attention:
                attention = SpatialAttention(filters)
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


class ModelValidator:
    """
    Comprehensive model validation and verification system.
    """

    @staticmethod
    def validate_configuration(config: Dict) -> Dict[str, Any]:
        """Validate model configuration parameters."""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }

        # Check input channels
        if config['input_channels'] != INPUT_CHANNELS:
            validation_results['warnings'].append(
                f"Non-standard input channels: {config['input_channels']}. Chess typically uses {INPUT_CHANNELS} channels."
            )

        # Check board size
        if config['board_size'] != 8:
            validation_results['errors'].append(
                f"Invalid board size: {config['board_size']}. Chess board must be 8x8."
            )
            validation_results['valid'] = False

        # Check conv filters progression
        conv_filters = config['conv_filters']
        if not all(conv_filters[i] <= conv_filters[i + 1] for i in range(len(conv_filters) - 1)):
            validation_results['warnings'].append(
                "Convolutional filters don't follow increasing pattern."
            )

        # Check dropout rate
        if not 0.0 <= config['dropout_rate'] <= 0.5:
            validation_results['warnings'].append(
                f"Dropout rate {config['dropout_rate']} may be too high or low. Recommended: 0.1-0.5"
            )

        # Check transformer configuration
        if config['use_transformer_blocks'] and config['num_transformer_layers'] > 4:
            validation_results['warnings'].append(
                "Too many transformer layers may cause overfitting."
            )

        return validation_results

    @staticmethod
    def test_forward_pass(model: nn.Module, input_shape: Tuple[int, ...], device: str = 'cpu') -> Dict[str, Any]:
        """Test model forward pass with dummy data."""
        test_results = {
            'success': False,
            'output_shape': None,
            'error': None,
            'memory_usage': None
        }

        try:
            model.eval()
            model.to(device)

            # Create dummy input
            dummy_input = torch.randn(input_shape).to(device)

            # Forward pass
            with torch.no_grad():
                output = model(dummy_input)

            test_results['success'] = True
            test_results['output_shape'] = tuple(output.shape)

            # Memory usage (approximate)
            if device == 'cuda':
                test_results['memory_usage'] = torch.cuda.memory_allocated() / 1024 ** 2  # MB

        except Exception as e:
            test_results['error'] = str(e)

        return test_results


class WandBIntegratedTrainer:
    """
    Enhanced trainer with comprehensive W&B integration, model summaries, and validation.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader,
            val_loader,
            config: Dict,
            project_name: str = "enhanced-chess-cnn",
            experiment_name: str = None,
            learning_rate: float = 0.001,
            weight_decay: float = 1e-4,
            scheduler_type: str = 'reduce_on_plateau',
            early_stopping_patience: int = 10
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Initialize W&B
        load_dotenv()
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            tags=["chess", "cnn", "enhanced", "pytorch"],
            notes="Enhanced Chess CNN with residual connections, attention, and transformers"
        )
        print("✓ Wandb initialized successfully")

        # Log model architecture to W&B
        wandb.watch(self.model, log_freq=100, log="all")

        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler
        if scheduler_type == 'exponential':
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        elif scheduler_type == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5,
                patience=5, min_lr=1e-7
            )
        else:
            self.scheduler = None

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

    def print_model_summary(self):
        """Print comprehensive model summary and log to W&B."""
        print("=" * 80)
        print("ENHANCED CHESS CNN MODEL SUMMARY")
        print("=" * 80)

        # Configuration summary
        print("\n📋 MODEL CONFIGURATION:")
        print("-" * 40)
        for key, value in self.config.items():
            print(f"{key:25s}: {value}")

        # Parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n📊 PARAMETER STATISTICS:")
        print("-" * 40)
        print(f"{'Total Parameters':<25s}: {total_params:,}")
        print(f"{'Trainable Parameters':<25s}: {trainable_params:,}")
        print(f"{'Non-trainable Parameters':<25s}: {total_params - trainable_params:,}")

        # Model architecture using torchinfo
        print(f"\n🏗️  DETAILED ARCHITECTURE:")
        print("-" * 40)
        try:
            model_summary = summary(
                self.model,
                input_size=(1, self.config['input_channels'],
                            self.config['board_size'], self.config['board_size']),
                verbose=1,
                col_names=["output_size", "num_params", "mult_adds"],
                row_settings=["depth"]
            )

            # Log summary to W&B
            wandb.log({
                "model/total_params": total_params,
                "model/trainable_params": trainable_params,
                "model/model_size_mb": total_params * 4 / (1024 ** 2)  # Assuming float32
            })

        except Exception as e:
            print(f"Could not generate detailed summary: {e}")

        print("=" * 80)

    def validate_model_setup(self):
        """Comprehensive model validation and configuration checking."""
        print("\n🔍 MODEL VALIDATION AND VERIFICATION")
        print("=" * 80)

        # Validate configuration
        validator = ModelValidator()
        validation_results = validator.validate_configuration(self.config)

        print("📋 Configuration Validation:")
        print("-" * 40)
        if validation_results['valid']:
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration has errors:")
            for error in validation_results['errors']:
                print(f"   • {error}")

        if validation_results['warnings']:
            print("⚠️  Warnings:")
            for warning in validation_results['warnings']:
                print(f"   • {warning}")

        if validation_results['recommendations']:
            print("💡 Recommendations:")
            for rec in validation_results['recommendations']:
                print(f"   • {rec}")

        # Test forward pass
        print("\n🚀 Forward Pass Test:")
        print("-" * 40)
        device = next(self.model.parameters()).device
        input_shape = (2, self.config['input_channels'],
                       self.config['board_size'], self.config['board_size'])

        test_results = validator.test_forward_pass(self.model, input_shape, str(device))

        if test_results['success']:
            print("✅ Forward pass successful")
            print(f"   • Input shape: {input_shape}")
            print(f"   • Output shape: {test_results['output_shape']}")
            if test_results['memory_usage']:
                print(f"   • Memory usage: {test_results['memory_usage']:.2f} MB")
        else:
            print("❌ Forward pass failed:")
            print(f"   • Error: {test_results['error']}")

        # Log validation results to W&B
        wandb.log({
            "validation/config_valid": validation_results['valid'],
            "validation/num_warnings": len(validation_results['warnings']),
            "validation/num_errors": len(validation_results['errors']),
            "validation/forward_pass_success": test_results['success']
        })

        print("=" * 80)

    def log_training_step(self, epoch: int, train_loss: float, val_loss: float,
                          current_lr: float, step: int):
        """Log training metrics to W&B."""
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/learning_rate": current_lr,
            "validation/loss": val_loss,
            "step": step
        }, step=step)

    def train_epoch(self):
        """Train for one epoch with gradient clipping and W&B logging."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Log batch-level metrics occasionally
            if batch_idx % 100 == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch_idx": batch_idx
                })

        return total_loss / num_batches

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, num_epochs: int):
        """Complete training loop with W&B integration and comprehensive logging."""
        print("\n🚀 STARTING ENHANCED TRAINING")
        print("=" * 80)
        print(f"Training Configuration:")
        print(f"• Epochs: {num_epochs}")
        print(f"• Optimizer: {type(self.optimizer).__name__}")
        print(f"• Learning Rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"• Weight Decay: {self.optimizer.param_groups[0]['weight_decay']}")
        print(f"• Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
        print(f"• Early Stopping Patience: {self.early_stopping.patience}")
        print("=" * 80)

        global_step = 0

        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch()

            # Validation
            val_loss = self.validate()

            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            # Log to W&B
            self.log_training_step(epoch, train_loss, val_loss, current_lr, global_step)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
                wandb.log({"training/early_stopped": True, "training/early_stop_epoch": epoch + 1})
                break

            # Progress reporting
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"📊 Epoch {epoch + 1:3d}/{num_epochs}")
                print(f"   • Train Loss: {train_loss:.6f}")
                print(f"   • Val Loss:   {val_loss:.6f}")
                print(f"   • LR:         {current_lr:.2e}")
                print("-" * 40)

            global_step += 1

        # Final model save to W&B
        torch.save(self.model.state_dict(), "enhanced_chess_cnn_final.pth")
        wandb.save("enhanced_chess_cnn_final.pth")

        # Log final metrics
        wandb.log({
            "training/final_train_loss": self.train_losses[-1],
            "training/final_val_loss": self.val_losses[-1],
            "training/total_epochs": len(self.train_losses)
        })

        print(f"\n✅ Training completed!")
        print(f"Final train loss: {self.train_losses[-1]:.6f}")
        print(f"Final validation loss: {self.val_losses[-1]:.6f}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }


# Updated early stopping class (keeping from original)
class EarlyStopping:
    """Early stopping implementation to prevent overfitting."""

    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


def create_enhanced_chess_model_with_validation(config=None):
    """
    Factory function to create enhanced chess model with comprehensive validation.
    """
    if config is None:
        config = {
            'input_channels': INPUT_CHANNELS,
            'board_size': 8,
            'conv_filters': [64, 128, 256],
            'fc_layers': [512, 256],
            'dropout_rate': 0.3,
            'batch_norm': True,
            'activation': 'gelu',
            'use_attention': True,
            'use_transformer_blocks': True,
            'num_transformer_layers': 2,
            'transformer_heads': 8
        }

    print("🏗️  CREATING ENHANCED CHESS CNN MODEL")
    print("=" * 80)

    # Create model
    model = EnhancedChessCNN(**config)

    # Validate configuration
    validator = ModelValidator()
    validation_results = validator.validate_configuration(config)

    if not validation_results['valid']:
        raise ValueError(f"Invalid configuration: {validation_results['errors']}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✅ Enhanced Chess CNN created successfully!")
    print(f"📊 Total Parameters: {total_params:,}")
    print(f"💾 Estimated Model Size: {total_params * 4 / (1024 ** 2):.2f} MB")

    if validation_results['warnings']:
        print(f"⚠️  {len(validation_results['warnings'])} warnings found")

    print("=" * 80)

    return model, config


# Example usage with comprehensive setup
if __name__ == "__main__":
    # Install required packages (uncomment if needed)
    # !pip install wandb torchinfo

    print("🎯 ENHANCED CHESS CNN WITH W&B INTEGRATION")
    print("=" * 80)

    # Create enhanced model with validation
    model, config = create_enhanced_chess_model_with_validation()

    # Example training setup (requires actual data loaders)
    # Note: You'll need to replace these with actual data loaders
    train_loader = None  # Your training data loader
    val_loader = None  # Your validation data loader

    # Initialize trainer with W&B integration
    trainer = WandBIntegratedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        project_name="enhanced-chess-cnn",
        experiment_name="residual-attention-transformer",
        learning_rate=0.001,
        weight_decay=1e-4,
        scheduler_type='reduce_on_plateau',
        early_stopping_patience=15
    )

    # Print comprehensive model summary
    trainer.print_model_summary()

    # Validate model setup
    trainer.validate_model_setup()

    if train_loader is not None and val_loader is not None:
        # Train the model
        training_history = trainer.train(num_epochs=200)
        # Close W&B run
        wandb.finish()
    else:
        print("⚠️  Training data loaders not provided. Model created but not trained.")
        print("   To train, provide train_loader and val_loader parameters.")

    print("\n✅ Enhanced Chess CNN setup completed!")
