import platform
import time
import warnings

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
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
            dataset: list[Dataset],
            train_loaders,
            val_loader,
            dataset_rootname: str,
            project_name: str = "chess-cnn",
            experiment_name: str = None,
            scheduler_type: str = 'reduce_on_plateau',
            early_stopping_patience: int = 10
    ):
        self.model = model
        self.device = next(self.model.parameters()).device
        if not str(self.device).__contains__(TRAINING_CONFIG["device"]):
            raise Exception(f"next(self.model.parameters()).device should be {TRAINING_CONFIG["device"]} but is {self.device}")
        print(f"✓ Training on CUDA with {len(train_loaders)} training datasets")

        self.dataset_rootname = dataset_rootname
        self.train_loaders = train_loaders
        self.dataset = dataset
        self.val_loader = val_loader
        self.batch_logger = EfficientBatchLogger(log_frequency=25)
        # Gradient accumulation settings
        self.accumulation_steps = TRAINING_CONFIG["accumulation_steps"]
        self.effective_batch_size = TRAINING_CONFIG["batch_size"] * self.accumulation_steps

        # Initialize W&B
        load_dotenv()
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=TRAINING_CONFIG["config"],
            tags=["chess", "cnn", "enhanced", "pytorch"],
            notes="Enhanced Chess CNN with residual connections, attention, and transformers"
        )
        print("✓ Wandb initialized successfully")

        # Log model architecture to W&B
        wandb.watch(self.model, log_freq=100, log="all")
        self.scaler = GradScaler()  # Add this line for AMP

        if TRAINING_CONFIG["with_ema"]:
            self.ema = EMA(model, decay=0.999)

        attention_params = []
        other_params = []

        for name, param in model.named_parameters():
            if 'attention' in name:
                attention_params.append(param)
            else:
                other_params.append(param)

        self.optimizer  = torch.optim.AdamW([
            {'params': attention_params, 'lr': TRAINING_CONFIG["learning_rate"], 'weight_decay': TRAINING_CONFIG["attention"]["weight_decay"]},
            {'params': other_params, 'lr': TRAINING_CONFIG["attention"]["learning_rate"], 'weight_decay': TRAINING_CONFIG["weight_decay"]},
        ])


        # Optimizer with weight decay (L2 regularization)
        # self.optimizer = optim.AdamW(
        #     model.parameters(),
        #     lr=TRAINING_CONFIG["learning_rate"],
        #     weight_decay=TRAINING_CONFIG["weight_decay"],
        #     betas=(0.9, 0.999),
        #     eps=1e-8,
        #     fused=True  # PyTorch 2.0+ fused optimizer
        # )

        self.scaler = GradScaler() #For AMP
        self.criterion = nn.CrossEntropyLoss()  # Simple, fast loss
        # Wrap optimizer with gradient noise, but too heavy
        # self.optimizer = GradientNoiseOptimizer(optimizer, noise_std=0.01, decay=0.55)
        # Initialize focal loss, but too heavy
        # self.criterion = FocalLoss(alpha=0.25, gamma=2.0)

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
        elif scheduler_type == 'cosine_annealing':
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=TRAINING_CONFIG["cosine"]["warmup_epochs"])
            cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=TRAINING_CONFIG["num_epoch"] - TRAINING_CONFIG["cosine"]["warmup_epochs"])
            self.scheduler = SequentialLR(self.optimizer, [warmup_scheduler, cosine_scheduler], [TRAINING_CONFIG["cosine"]["warmup_epochs"]])
        elif scheduler_type == 'cosine_annealing_warm_restarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=TRAINING_CONFIG["cosine"]["first_restart"],  # First restart after 10 epochs
                T_mult=2,  # Double the cycle length after each restart
                eta_min=TRAINING_CONFIG["cosine"]["eta_min"]  # Minimum learning rate (adjust as needed)
            )
        else:
            self.scheduler = None
        print(f"✓ Trainer initialized - Effective batch size: {self.effective_batch_size}")

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
        for key, value in TRAINING_CONFIG["config"].items():
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
            summary(
                self.model,
                input_size=(1, TRAINING_CONFIG['input_channels'],
                            TRAINING_CONFIG['board_size'], TRAINING_CONFIG['board_size']),
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
        ModelValidator().validate_all(self.model)

    def _cleanup_epoch(self, cycle):
        """Release resources after each epoch"""
        if hasattr(self.train_loaders[cycle], 'clear_cache'):
            self.train_loaders[cycle].clear_cache()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(self.train_loaders[cycle], '_inputs'):
            os.close(self.train_loaders[cycle]._inputs)
        if hasattr(self.train_loaders[cycle], '_outputs'):
            os.close(self.train_loaders[cycle]._outputs)
        if not platform.system() == 'Windows':
            self._release_linux_resources()

    def _cleanup_validation(self):
        """Release resources after each epoch"""
        if hasattr(self.val_loader, 'clear_cache'):
            self.val_loader.clear_cache()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(self.val_loader, '_inputs'):
            os.close(self.val_loader._inputs)
        if hasattr(self.val_loader, '_outputs'):
            os.close(self.val_loader._outputs)
        if not platform.system() == 'Windows':
            self._release_linux_resources()

    def _release_linux_resources(self):
        """Release Linux-specific resources"""
        # 1. Clear page cache (requires sudo)
        os.system('sync; echo 1 > /proc/sys/vm/drop_caches')

        # 2. Release memory-mapped files
        os.system(f'fuser -k {self.dataset_rootname}*.dat')  # Kill processes

    def train_epoch(self, epoch):
        # Get model device dynamically
        start_time = time.time()
        cycle = epoch % len(self.train_loaders)
        print(f"[{start_time:0f}] start training epoch : {epoch}, cycle {cycle}")
        self.model = self.model.cuda()  # Redundant but safe
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        total_data = 0
        num_updates = 0
        self.optimizer.zero_grad(set_to_none=True)
        trainer = self.train_loaders[cycle]
        for batch_idx, (data, target) in enumerate(trainer):
            num_batches += 1
            # CRITICAL: Mark beginning of each iteration,
            # otherwise: RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run.
            # on: self.scaler.scale(loss).backward()
            torch.compiler.cudagraph_mark_step_begin()
            # with torch.cuda.stream(torch.cuda.Stream()):
            batch_start_time = time.time()
            total_data += len(data)
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True).argmax(dim=1)  # Convert one-hot
            # Apply mixup augmentation, but too heavy
            if TRAINING_CONFIG["with_mixup"]:
                data, targets_a, targets_b, lam = mixup_data(data, target, alpha=0.1)
            # Forward pass with AMP
            with autocast(device_type='cuda', dtype=torch.float16):
                output = self.model(data)
                # More precise? More complex
                if TRAINING_CONFIG["with_mixup"]:
                    loss = mixup_criterion(self.criterion, output, targets_a, targets_b, lam)
                else:
                    loss = self.criterion(output, target) / self.accumulation_steps


            # Backward pass
            self.scaler.scale(loss).backward()
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)

                # Apply gradient centralization
                # Gradient centralization improves training stability by removing the mean from gradients . This reduces internal covariate shift and helps with faster convergence.
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.data = centralize_gradient(param.grad.data)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=TRAINING_CONFIG["gradient_clipping"])

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                # Update EMA parameters, but heavy
                if TRAINING_CONFIG["with_ema"]:
                    self.ema.update()
                num_updates += 1

            total_loss += loss.item() * self.accumulation_steps
            # Minimal logging (every 100 batches)
            batch_time = time.time() - batch_start_time
            if batch_idx % 100 == 0:
                throughput = (batch_idx + 1) * TRAINING_CONFIG["batch_size"] / (time.time() - start_time)
                if hasattr(self.dataset[cycle + 1], 'get_cache_stats'):
                    stats = self.dataset[cycle + 1].get_cache_stats()
                    print(
                        f"Batch {batch_idx:5d} | Loss: {loss.item():.4f} | "
                        f"Throughput: {throughput:.2f} samples/sec | "
                        f"Cache hit rate: {stats['hit_rate']:.2%} | "
                        f"Cache utilization: {stats['cache_size']}/{stats['capacity']} | "
                    )
                else:
                    print(f"Batch {batch_idx:5d} | Loss: {loss.item() * self.accumulation_steps:.4f} | "
                        f"Throughput: {throughput:.2f} samples/sec")

                self.batch_logger.log_batch_metrics(
                    batch_idx,
                    loss.item(),
                    0.0,
                    self.optimizer.param_groups[0]['lr']
                )
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch_idx": batch_idx,
                    "train/epoch_throughput": throughput,
                })
                self.batch_logger.log_detailed_batch_info(self.model, trainer, batch_idx, epoch, batch_time, loss.item())

        stop = time.time()
        if num_batches >0:
            print(f"[{stop}] epoch {epoch} done processing {total_data} positions in {num_batches} batches in {stop - start_time:.2f} seconds ({total_data / num_batches:.2f} ==? {TRAINING_CONFIG["batch_size"]})")
        return total_loss / num_batches if num_batches > 0 else 0

    def validate(self):
        """Validate the model."""
        self.model.eval()
        # Update EMA parameters, but heavy
        if TRAINING_CONFIG["with_ema"]:
            self.ema.apply_shadow()
        total_loss = 0.0

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True).argmax(dim=1)
                with autocast(device_type='cuda', dtype=torch.float16):
                    output = self.model(data)
                    loss = self.criterion(output, target)
                total_loss += loss.item()

        # Restore original parameters for next training epoch
        if TRAINING_CONFIG["with_ema"]:
            self.ema.restore()
        torch.cuda.empty_cache()
        return total_loss / len(self.val_loader)

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
            self._cleanup_epoch(epoch)
            # Validation
            if epoch % len(self.train_loaders) == len(self.train_loaders) - 1:
                val_loss = self.validate()
                self._cleanup_validation()

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