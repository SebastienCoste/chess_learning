import wandb
import torch

class EfficientBatchLogger:
    def __init__(self, log_frequency=50):
        self.log_frequency = log_frequency
        self.batch_metrics = []
        self.running_loss = 0.0
        self.batch_count = 0

    def log_batch_metrics(self, batch_idx, loss, accuracy, learning_rate):
        """Accumulate batch metrics for efficient logging"""
        self.running_loss += loss
        self.batch_count += 1

        # Log detailed metrics every N batches
        if batch_idx % self.log_frequency == 0:
            avg_loss = self.running_loss / self.batch_count

            wandb.log({
                "batch/loss": loss,
                "batch/running_avg_loss": avg_loss,
                "batch/accuracy": accuracy,
                "batch/learning_rate": learning_rate,
                "batch/batch_idx": batch_idx
            }, commit=False)

            # Reset accumulators
            self.running_loss = 0.0
            self.batch_count = 0

    def log_detailed_batch_info(self, model, train_loader, batch_idx, epoch, batch_time, loss):
        """Log detailed batch information periodically"""

        # Memory usage tracking
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            gpu_cached = torch.cuda.memory_reserved() / 1024 ** 3  # GB
        else:
            gpu_memory = gpu_cached = 0.0

        # Gradient norms for monitoring training stability
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        # Comprehensive logging
        wandb.log({
            "detailed/epoch": epoch,
            "detailed/batch_time": batch_time,
            "detailed/gpu_memory_gb": gpu_memory,
            "detailed/gpu_cached_gb": gpu_cached,
            "detailed/gradient_norm": total_norm,
            "detailed/samples_per_second": len(train_loader.dataset) / batch_time,
        }, commit=False)

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
        self.batch_metrics = []
        self.running_loss = 0.0
        self.batch_count = 0