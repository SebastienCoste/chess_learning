import time

import torch
import torch.nn as nn
import torch.optim as optim
from setuptools.errors import InvalidConfigError
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import wandb
from torchinfo import summary
from typing import Dict
import os
from dotenv import load_dotenv
import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch.amp import autocast, GradScaler

from cnn.chess_v3.components.WandbLogger import EfficientBatchLogger
from cnn.chess_v3.components.config import TRAINING_CONFIG
from generate_data import create_chess_data_loaders
from cnn.chess_v3.components.model_validator import ModelValidator
from cnn.chess_v3.components.early_stopping import EarlyStopping
from cnn.chess_v3.components.chess_cnn import EnhancedChessCNN


class Trainer:
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
        self.scaler = GradScaler()  # Add this line for AMP

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
                self.optimizer, mode='min', factor=0.2,
                patience=5, min_lr=1e-7
            )
        elif scheduler_type == 'cosine':
            self.scheduler = LinearWarmupCosineAnnealingLR(
                self.optimizer,
                warmup_epochs=TRAINING_CONFIG["cosine"]["warmup_epochs"],
                max_epochs=200,
                warmup_start_lr=0.0001,
                eta_min=TRAINING_CONFIG["cosine"]["eta_min"]
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
            "validation/forward_pass_success": test_results['success'],
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
        })

    def train_epoch(self, epoch):
        """Train for one epoch with gradient clipping and W&B logging."""
        self.model.train()
        if TRAINING_CONFIG["device"] == "cuda":
            self.model = self.model.to('cuda')  # Redundant but safe
        total_loss = 0.0
        num_batches = 0

        batch_logger = EfficientBatchLogger(log_frequency=25)

        # Get model device dynamically
        device = next(self.model.parameters()).device

        if not str(device).__contains__(TRAINING_CONFIG["device"]):
            raise Exception(f"next(self.model.parameters()).device should be {TRAINING_CONFIG["device"]} but is {device}")

        for batch_idx, (data, target) in enumerate(self.train_loader):
            batch_start_time = time.time()
            if TRAINING_CONFIG["device"] == "cuda":
                # Data should be already on CUDA from DataLoader
                data = data.to(device)
                target = target.to(device)
            if not str(data.device).__contains__(TRAINING_CONFIG["device"]):
                print(f"Data device not {TRAINING_CONFIG["device"]}: {data.device}")  # Should be cuda:0
            if not str(target.device).__contains__(TRAINING_CONFIG["device"]):
                print(f"Target device not {TRAINING_CONFIG["device"]}: {target.device}")  # Should be cuda:0
            if not str(next(model.parameters()).device).__contains__(TRAINING_CONFIG["device"]):
                print(f"Model device not {TRAINING_CONFIG["device"]}: {next(model.parameters()).device}")  # Should be cuda:0
            self.optimizer.zero_grad()
            if TRAINING_CONFIG["mixed_precision"]:
                # --- AMP block starts here ---
                with autocast(dtype=torch.float16, device_type="cuda"):
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Unscale gradients before clipping (recommended)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)

                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Scheduler step (if not ReduceLROnPlateau)
            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            else:
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            batch_time = time.time() - batch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']

            if batch_idx % 10 == 1:  # Calculate accuracy every 10 batches
                with torch.no_grad():
                    pred = output.argmax(dim=1, keepdim=True)
                    print(f"pred shape: {pred.shape}, target shape: {target.shape}")
                    pred = output.argmax(dim=1)  # [batch_size, H, W]
                    correct = pred.eq(target).sum().item()
                    # correct = pred.eq(target.view_as(pred)).sum().item()
                    accuracy = 100. * correct / len(data)
            else:
                accuracy = None

                # Log batch metrics efficiently
            batch_logger.log_batch_metrics(
                batch_idx,
                loss.item(),
                accuracy if accuracy else 0.0,
                current_lr
            )

            # Additional detailed logging for specific batches
            if batch_idx % 25 == 0:
                batch_logger.log_detailed_batch_info(self.model, self.train_loader, batch_idx, epoch, batch_time, loss.item())
            # Log batch-level metrics occasionally
            # if batch_idx % 25 == 0 or batch_idx == 0:
            #     wandb.log({
            #         "train/batch_loss": loss.item(),
            #         "train/batch_idx": batch_idx
            #     })

        return total_loss / num_batches

    def validate(self):
        """Validate the model."""
        self.model.eval()
        device = next(self.model.parameters()).device
        if not str(device).__contains__(TRAINING_CONFIG["device"]):
            raise Exception(f"next(self.model.parameters()).device is {device} but should be {TRAINING_CONFIG['device']}")
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(device)
                target = target.to(device)
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
            train_loss = self.train_epoch(epoch)

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
                wandb.log({"training/early_stopped": True, "training/early_stop_epoch": epoch + 1}, commit=True)
                break

            # Progress reporting
            if True or (epoch + 1) % 10 == 0 or epoch == 0:
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

# Load processed training data
def load_chess_training_data(filename):
    with open(filename, 'rb') as f:
        training_data = pickle.load(f)
    return training_data

def create_enhanced_chess_model_with_validation(config=None):
    """
    Factory function to create enhanced chess model with comprehensive validation.
    """
    if config is None:
        config = TRAINING_CONFIG["config"]

    print("🏗️  CREATING ENHANCED CHESS CNN MODEL")
    print("=" * 80)

    # Create model
    if TRAINING_CONFIG["device"] == "cuda" and not torch.cuda.is_available():
        raise Exception("CUDA required in TRAINING_CONFIG but not available")
    model = EnhancedChessCNN(**config).to('cuda' if TRAINING_CONFIG["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"EnhancedChessCNN is using device {next(model.parameters()).device}")
    if TRAINING_CONFIG["device"] == "cuda":
        if not all(param.device.type == 'cuda' for param in model.parameters()):
            raise Exception(f"CUDA device not configured in a parameter but CUDA Expected.")
        if not all(buffer.device.type == 'cuda' for buffer in model.buffers()):
            raise Exception(f"CUDA device not configured in a buffer but CUDA Expected.")
        print(f"EnhancedChessCNN Parameters and buffers validated with CUDA")

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
    pkl_file = 'chess_training_data.pkl'
    # pkl_file = "minimal_train_data.pkl" #full training data: 'chess_training_data.pkl'

    print("🎯 ENHANCED CHESS CNN WITH W&B INTEGRATION")
    print("=" * 80)

    # Create enhanced model with validation
    model, config = create_enhanced_chess_model_with_validation()

    # Example training setup (requires actual data loaders)
    training_data = load_chess_training_data(pkl_file)
    print(f"Loaded training data from {pkl_file}")
    train_loader, val_loader = create_chess_data_loaders(
        training_data,
        train_split=0.8,
        batch_size= TRAINING_CONFIG["batch_size"],  # To be adjusted
        num_workers= TRAINING_CONFIG["num_workers"]  # 8 cores, 16 logical cores
    )

    # Initialize trainer with W&B integration
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        project_name=f"chess-cnn",
        experiment_name=f"run-{pkl_file.replace(".pkl", "")}-{TRAINING_CONFIG["version"]}",
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        scheduler_type=TRAINING_CONFIG["scheduler_type"],
        early_stopping_patience=TRAINING_CONFIG["early_stopping_patience"]
    )

    # Print comprehensive model summary
    trainer.print_model_summary()

    # Validate model setup
    trainer.validate_model_setup()

    if train_loader is not None and val_loader is not None:
        # Train the model
        training_history = trainer.train(num_epochs=TRAINING_CONFIG["num_epoch"])
        # Close W&B run
        wandb.finish()
    else:
        print("⚠️  Training data loaders not provided. Model created but not trained.")
        print("   To train, provide train_loader and val_loader parameters.")

    print("\n✅ Enhanced Chess CNN setup completed!")
