import sys
import os

# Critical: Set these BEFORE importing torch
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "ATEN"  # Disable Triton kernels
os.environ["TORCHINDUCTOR_AUTOTUNE_FALLBACK_TO_ATEN"] = "1"     # Fallback to ATEN when needed
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"         # Force recompilation with new settings
os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import platform
import warnings

import numpy as np
from pl_bolts.utils.stability import UnderReviewWarning

from cnn.chess.components.cnn.chess_cnn_v3 import EnhancedChessCNNV3
from cnn.chess.components.trainer import Trainer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UnderReviewWarning)
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import wandb
import os
import pickle

import multiprocessing as mp

import torch
from torch.utils.data import Dataset
# torch._dynamo.config.capture_scalar_outputs = True
import signal
import atexit

from cnn.chess.components.config import TRAINING_CONFIG
from cnn.chess.components.data_prep.mmap_dataset import MemmapChessDataset, MemmapChessDatasetWindows
from generate_data import create_optimized_dataloaders
from cnn.chess.components.training.model_validator import ModelValidator


# Load processed training data
def load_chess_training_data(filename):
    with open(filename, 'rb') as f:
        training_data = pickle.load(f)
    return training_data

def create_enhanced_chess_model_with_validation(config=TRAINING_CONFIG["config"]):
    """
    Factory function to create enhanced chess model with comprehensive validation.
    """
    print("🏗️  CREATING ENHANCED CHESS CNN MODEL")
    print("=" * 80)


    print("🔥 Compiling model with torch.compile...")
    is_windows = platform.system() == 'Windows'
    print(f"🔥 Compiling model with for windows? {is_windows}")
    uncompiled_model = EnhancedChessCNNV3(**config).cuda()
    if not is_windows:
        compiled_model = torch.compile(
            uncompiled_model,
            mode="default",        # Changed from reduce-overhead
            fullgraph=False,       # Changed from True
            dynamic=False
        )
        print("✓ Model compiled with fallback settings")
    else:
        compiled_model = uncompiled_model
        print("✓ Using uncompiled model on Windows")
    print("✓ Model compiled successfully")

    # Validate configuration
    validator = ModelValidator()
    validation_results = validator.validate(compiled_model)

    if not validation_results['valid']:
        raise ValueError(f"Invalid configuration: {validation_results['errors']}")

    # Count parameters
    total_params = sum(p.numel() for p in compiled_model.parameters() if p.requires_grad)

    print(f"✅ Enhanced Chess CNN created successfully!")
    print(f"📊 Total Parameters: {total_params:,}")
    print(f"💾 Estimated Model Size: {total_params * 4 / (1024 ** 2):.2f} MB")

    if validation_results['warnings']:
        print(f"⚠️  {len(validation_results['warnings'])} warnings found")

    print("=" * 80)

    return uncompiled_model, compiled_model, config

def handler(signum, frame):
    os._exit(1)

def cleanup():
    torch.distributed.destroy_process_group() if torch.distributed.is_initialized() else None
    torch.cuda.empty_cache()

# System-level optimizations
def optimize_system_settings():
    """Apply system-level optimizations for RTX 5080 + 128GB RAM"""

    # CUDA optimizations for RTX 5080
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    atexit.register(cleanup)

    # Memory optimizations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

    # CPU optimizations
    torch.set_num_threads(16)
    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['MKL_NUM_THREADS'] = '16'

    print("✓ System optimizations applied")


def create_trainer(m, ds_root, trains, validate, data: list[Dataset]):
    return Trainer(
        model=m,
        dataset=data,
        dataset_rootname = ds_root,
        train_loaders=trains,
        val_loader=validate,
        project_name=f"chess-cnn",
        experiment_name=f"run-{TRAINING_CONFIG["pth_file"]}-{TRAINING_CONFIG["version"]}",
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        scheduler_type=TRAINING_CONFIG["scheduler_type"],
        early_stopping_patience=TRAINING_CONFIG["early_stopping_patience"]
    )


# Example usage with comprehensive setup
if __name__ == "__main__":

    # Apply system optimizations
    optimize_system_settings()
    train_type = "all_train_data_with_puzzles_v3"
    mmap_file = f'data/{train_type}'
    split = 5

    print("🎯 ENHANCED CHESS CNN WITH W&B INTEGRATION")
    print("=" * 80)

    # Create enhanced model with validation
    uncompiled_model, model, config = create_enhanced_chess_model_with_validation()
    datasets = []
    for sub in range(split):
        is_windows = platform.system() == 'Windows'
        if is_windows:
            mp.set_start_method('spawn', force=True)
            datasets.append(MemmapChessDatasetWindows(f'data/{train_type}_{sub}'))
        else:
            datasets.append(MemmapChessDataset(f'data/{train_type}_{sub}'))

    print(f"Loaded training data from {mmap_file} on {platform.system()}")
    cached_datasets: list[Dataset]
    train_loaders, val_loader, cached_datasets = create_optimized_dataloaders(datasets, base_path=mmap_file, batch_size=TRAINING_CONFIG["batch_size"], cache_type=TRAINING_CONFIG["cache_type"])

    print(f"train_loaders has {len(train_loaders)} datasets")
    # Initialize trainer with W&B integration
    trainer = create_trainer(model, mmap_file, train_loaders, val_loader, cached_datasets)
    print(f"Model should be on CUDA: {next(model.parameters()).device}")

    # Print comprehensive model summary
    create_trainer(uncompiled_model, mmap_file, train_loaders, val_loader, cached_datasets).print_model_summary()

    # Validate model setup
    trainer.validate_model_setup()

    if train_loaders is not None and val_loader is not None:
        # Train the model
        training_history = trainer.train(num_epochs=TRAINING_CONFIG["num_epoch"])
        # Close W&B run
        wandb.finish()
    else:
        print("⚠️  Training data loaders not provided. Model created but not trained.")
        print("   To train, provide train_loader and val_loader parameters.")

    print("\n✅ Enhanced Chess CNN setup completed!")
