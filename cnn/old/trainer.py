import time

import torch.nn as nn
import torch.optim as optim

import numpy as np
import warnings

from pl_bolts.utils.stability import UnderReviewWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UnderReviewWarning)
if not hasattr(np, 'bool'):
    np.bool = np.bool_
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
import wandb
from torchinfo import summary
from typing import Dict
import os
from dotenv import load_dotenv
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

import torch
from torch.amp import autocast, GradScaler

from cnn.chess.components.cnn.data_manip.gradient_noise_optimizer import GradientNoiseOptimizer
from cnn.chess.components.training.EMA import EMA
from cnn.chess.components.utils.WandbLogger import EfficientBatchLogger
from cnn.chess.components.config import TRAINING_CONFIG
from cnn.chess.components.cnn.modules.focal_loss import FocalLoss
from cnn.chess.components.utils.module_utils import centralize_gradient, mixup_data, mixup_criterion
from cnn.chess.components.training.model_validator import ModelValidator
from cnn.chess.components.training.early_stopping import EarlyStopping


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
            project_name: str = "chess-cnn",
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
        self.batch_logger = EfficientBatchLogger(log_frequency=25)

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

        self.ema = EMA(model, decay=0.999)

        # Optimizer with weight decay (L2 regularization)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        # Wrap optimizer with gradient noise
        self.optimizer = GradientNoiseOptimizer(optimizer, noise_std=0.01, decay=0.55)

        # Initialize focal loss
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)

        # Learning rate scheduler
        if scheduler_type == 'exponential':
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        elif scheduler_type == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.2,
                patience=5, min_lr=1e-7
            )
        elif scheduler_type == 'cosine':
            # raise Exception("unstable")
            self.scheduler = LinearWarmupCosineAnnealingLR(
                self.optimizer,
                warmup_epochs=TRAINING_CONFIG["cosine"]["warmup_epochs"],
                max_epochs=200,
                warmup_start_lr=0.0001,
                eta_min=TRAINING_CONFIG["cosine"]["eta_min"]
            )
        elif scheduler_type == 'cosine_annealing':
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=TRAINING_CONFIG["cosine"]["warmup_epochs"])
            cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=TRAINING_CONFIG["num_epoch"] - TRAINING_CONFIG["cosine"]["warmup_epochs"])
            self.scheduler = SequentialLR(self.optimizer, [warmup_scheduler, cosine_scheduler], [TRAINING_CONFIG["cosine"]["warmup_epochs"]])
        elif scheduler_type == 'cosine_annealing_warm_restarts':
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=TRAINING_CONFIG["cosine"]["warmup_epochs"]
            )
            restart_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=TRAINING_CONFIG["cosine"]["first_restart"],  # First restart after 10 epochs
                T_mult=2,  # Double the cycle length after each restart
                eta_min=TRAINING_CONFIG["cosine"]["eta_min"]  # Minimum learning rate (adjust as needed)
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, restart_scheduler],
                milestones=[TRAINING_CONFIG["cosine"]["warmup_epochs"]]
            )
        else:
            self.scheduler = None

        # # Loss function with label smoothing
        # self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

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

    def train_epoch(self, epoch):
        # Get model device dynamically
        device = next(self.model.parameters()).device

        if not str(device).__contains__(TRAINING_CONFIG["device"]):
            raise Exception(f"next(self.model.parameters()).device should be {TRAINING_CONFIG["device"]} but is {device}")
        start = time.time()
        print(f"[{start}] start training epoch : {epoch} on {device}")
        """Train for one epoch with gradient clipping and W&B logging."""
        if TRAINING_CONFIG["device"] == "cuda" and not str(device).__contains__("cuda"):
            self.model = self.model.to('cuda')  # Redundant but safe
        self.model.train()
        # print(f"self.model.train() done on epoch : {epoch} on {device} in {time.time() - start} seconds")
        total_loss = 0.0
        num_batches = 0
        # Get a single batch from the train_loader
        # d, t = next(iter(train_loader))

        # Check device for input and target
        # print(f"epoch {epoch}: Data {len(d)} device: {d.device}")
        # print(f"epoch {epoch}: Target {len(t)} device: {t.device}")
        total_data = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # print(f"start batch {batch_idx} of size {len(data)} for epoch : {epoch} on {device}")
            batch_start_time = time.time()
            total_data += len(data)
            if TRAINING_CONFIG["device"] == "cuda":
                # print(f"Data might already be on CUDA: {data.device}")
                # print(f"Target might already be on CUDA: {target.device}")
                # Data should be already on CUDA from DataLoader
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                # print(f"Data should be on CUDA: {data.device}")
                # print(f"Target should be on CUDA: {target.device}")
            if not str(data.device).__contains__(TRAINING_CONFIG["device"]):
                print(f"Data device not {TRAINING_CONFIG["device"]}: {data.device}")  # Should be cuda:0
            if not str(target.device).__contains__(TRAINING_CONFIG["device"]):
                print(f"Target device not {TRAINING_CONFIG["device"]}: {target.device}")  # Should be cuda:0
            if not str(next(model.parameters()).device).__contains__(TRAINING_CONFIG["device"]):
                print(f"Model device not {TRAINING_CONFIG["device"]}: {next(model.parameters()).device}")  # Should be cuda:0

            with torch.cuda.stream(torch.cuda.Stream()):
                # Apply mixup augmentation
                mixed_data, targets_a, targets_b, lam = mixup_data(data, target, alpha=0.2)

                if TRAINING_CONFIG["mixed_precision"]:
                    # --- AMP block starts here ---
                    with autocast(dtype=torch.float16, device_type="cuda"):
                        output = model(mixed_data)
                        loss = mixup_criterion(self.criterion, output, targets_a, targets_b, lam)
                        # Backward pass with gradient scaling
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()

                    # Unscale gradients before clipping (recommended)
                    self.scaler.unscale_(self.optimizer)
                    # Apply gradient centralization
                    #Gradient centralization improves training stability by removing the mean from gradients . This reduces internal covariate shift and helps with faster convergence.
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data = centralize_gradient(param.grad.data)

                    # Aggressive gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=TRAINING_CONFIG["gradient_clipping"]) #Reduced from 1.0

                    # Optimizer step with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    # Update EMA parameters
                    self.ema.update()
                else:
                    # Forward pass
                    output = self.model(data)
                    loss = self.criterion(output, target)

                    self.optimizer.zero_grad()
                    # Backward pass with gradient clipping
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=TRAINING_CONFIG["gradient_clipping"]) #Reduced from 1.0

                    # Scheduler step (if not ReduceLROnPlateau)
                    # if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    #     self.scheduler.step()
                    # else:
                    self.optimizer.step()

                    #Gradient centralization improves training stability by removing the mean from gradients . This reduces internal covariate shift and helps with faster convergence.
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data = centralize_gradient(param.grad.data)

                total_loss += loss.item()
                num_batches += 1

                batch_time = time.time() - batch_start_time
                current_lr = self.optimizer.param_groups[0]['lr']

                # if batch_idx % 10 == 1:  # Calculate accuracy every 10 batches
                #     with torch.no_grad():
                #         pred = output.argmax(dim=1, keepdim=True)
                #         print(f"pred shape: {pred.shape}, target shape: {target.shape}")
                #         pred = output.argmax(dim=1)  # [batch_size, H, W]
                #         correct = pred.eq(target).sum().item()
                #         # correct = pred.eq(target.view_as(pred)).sum().item()
                #         accuracy = 100. * correct / len(data)
                # else:
                accuracy = None

                self.optimizer.zero_grad(set_to_none=True)
                    # Log batch metrics efficiently
                self.batch_logger.log_batch_metrics(
                    batch_idx,
                    loss.item(),
                    accuracy if accuracy else 0.0,
                    current_lr
                )

                # Additional detailed logging for specific batches
                if batch_idx % 10 == 1:
                    self.batch_logger.log_detailed_batch_info(self.model, self.train_loader, batch_idx, epoch, batch_time, loss.item())
                # Log batch-level metrics occasionally
                if batch_idx % 25 == 0 or batch_idx == 0:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/batch_idx": batch_idx
                    })
        stop = time.time()
        print(f"[{stop}] epoch {epoch} done processing {total_data} positions in {num_batches} batches in {stop - start:.2f} seconds ({total_data / num_batches:.2f} ==? {TRAINING_CONFIG["batch_size"]})")
        return total_loss / num_batches

    def validate(self):
        """Validate the model."""
        self.model.eval()
        self.ema.apply_shadow()
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

        # Restore original parameters for next training epoch
        self.ema.restore()
        torch.cuda.empty_cache()
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
        print(f"• Config: {TRAINING_CONFIG["config"]}")
        print("=" * 80)

        global_step = 0
        torch.backends.cudnn.benchmark = True
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Training
            start = time.time()
            train_loss = self.train_epoch(epoch)

            # Validation
            val_loss = self.validate()

            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            # Log to W&B
            self.batch_logger.log_training_step(epoch, train_loss, val_loss, current_lr, global_step)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Checkpoint saving logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save model checkpoint
                checkpoint_path = f"models/{TRAINING_CONFIG["pth_file"]}_{TRAINING_CONFIG["version"]}_cp{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)

                print(f"\n💾 Saved new best model at epoch {epoch} with val loss {val_loss:.6f}")
                wandb.save(checkpoint_path)

                # Optional: Also log to W&B
                wandb.log({"best_val_loss": val_loss, "best_epoch": epoch})

            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                wandb.log({"training/early_stopped": True, "training/early_stop_epoch": epoch + 1}, commit=True)
                break


            print(f"📊 Epoch {epoch:3d}/{num_epochs}")
            print(f"   • Train Loss: {train_loss:.6f}")
            print(f"   • Val Loss:   {val_loss:.6f}")
            print(f"   • LR:         {current_lr:.2e}")
            print(f"   • Duration:   {time.time() - start:.2f} seconds")
            print("-" * 40)

            global_step += 1

        # Final model save to W&B
        torch.save(self.model.state_dict(), f"models/{TRAINING_CONFIG["pth_file"]}_{TRAINING_CONFIG["version"]}_final.pth")
        wandb.save(f"{TRAINING_CONFIG["pth_file"]}_{TRAINING_CONFIG["version"]}_final.pth")

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