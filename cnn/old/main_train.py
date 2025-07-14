#!/usr/bin/env python3
"""
Chess CNN Training Script

This script trains a Convolutional Neural Network to play chess by learning
from grandmaster games. It includes comprehensive logging with Weights & Biases,
data augmentation, and supports both fast and optimal training configurations.

Usage:
    python main_train.py --config fast --data_path ./data/sample_games.pgn
    python main_train.py --config optimal --wandb_project my-chess-project
"""

import argparse
import sys
import os
from pathlib import Path
import time
import torch

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def print_parameter_info():
    """Print detailed parameter information for user reference."""
    print("\n" + "="*80)
    print("CHESS CNN TRAINING PARAMETERS")
    print("="*80)
    
    parameter_explanations = {
        "learning_rate": {
            "description": "Controls how big steps the optimizer takes during training",
            "typical_range": "0.0001 to 0.01",
            "recommendations": {
                "fast_training": 0.002,
                "optimal_training": 0.001,
                "large_dataset": 0.0005
            },
            "effects": {
                "too_high": "Training instability, divergence",
                "too_low": "Very slow convergence, training stagnation"
            }
        },
        "batch_size": {
            "description": "Number of samples processed together in one forward pass",
            "typical_range": "16 to 512",
            "recommendations": {
                "limited_memory": 32,
                "standard": 64,
                "fast_training": 128
            },
            "effects": {
                "larger": "More stable gradients, faster training, more memory usage",
                "smaller": "Less memory usage, more gradient noise, potentially better generalization"
            }
        },
        "dropout_rate": {
            "description": "Fraction of neurons randomly set to zero during training",
            "typical_range": "0.0 to 0.5",
            "recommendations": {
                "small_dataset": 0.5,
                "standard": 0.3,
                "large_dataset": 0.1
            },
            "effects": {
                "higher": "More regularization, reduced overfitting, slower learning",
                "lower": "Less regularization, potential overfitting, faster learning"
            }
        },
        "weight_decay": {
            "description": "L2 regularization strength - penalizes large weights",
            "typical_range": "1e-6 to 1e-2",
            "recommendations": {
                "small_dataset": 1e-3,
                "standard": 1e-4,
                "large_dataset": 1e-5
            },
            "effects": {
                "higher": "Stronger regularization, smaller weights, reduced overfitting",
                "lower": "Weaker regularization, potential overfitting"
            }
        }
    }
    
    for param_name, info in parameter_explanations.items():
        print(f"\n{param_name.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Typical Range: {info['typical_range']}")
        print("  Recommendations:")
        for scenario, value in info['recommendations'].items():
            print(f"    - {scenario}: {value}")
        print("  Effects:")
        for effect, description in info['effects'].items():
            print(f"    - {effect}: {description}")
    
    print("\n" + "="*80)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a CNN to play chess",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training (< 8 hours)
  python main_train.py --config fast --data_path data/sample_games.pgn
  
  # Optimal training for best performance
  python main_train.py --config optimal --download_data
  
  # Custom configuration
  python main_train.py --lr 0.002 --batch_size 128 --epochs 50
  
  # Show parameter information
  python main_train.py --show_params
        """
    )
    
    # Configuration options
    parser.add_argument("--config", type=str, choices=["fast", "optimal", "debug"],
                       help="Use predefined configuration")
    parser.add_argument("--show_params", action="store_true",
                       help="Show detailed parameter information and exit")
    
    # Data options
    parser.add_argument("--data_path", type=str,
                       help="Path to PGN file with chess games")
    parser.add_argument("--download_data", action="store_true",
                       help="Download sample chess data automatically")
    parser.add_argument("--min_elo", type=int, default=2000,
                       help="Minimum ELO rating for games to include")
    parser.add_argument("--max_games", type=int,
                       help="Maximum number of games to process")
    
    # Model options
    parser.add_argument("--param_target_min", type=int, default=500000,
                       help="Minimum number of model parameters")
    parser.add_argument("--param_target_max", type=int, default=2000000,
                       help="Maximum number of model parameters")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    
    # Hardware options
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu", "mps"],
                       default="auto", help="Device to use for training")
    parser.add_argument("--mixed_precision", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--num_workers", type=int, help="Number of data loading workers")
    
    # WandB options
    parser.add_argument("--wandb_project", type=str, default="chess-cnn",
                       help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, help="WandB entity name")
    parser.add_argument("--wandb_run_name", type=str, help="WandB run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Directory to save model checkpoints")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()
    
    if args.show_params:
        print_parameter_info()
        return
    
    print("="*80)
    print("CHESS CNN TRAINING")
    print("="*80)
    print("For a complete implementation, please:")
    print("1. Create the full project structure as shown in README.md")
    print("2. Implement all the required modules (config, models, data, training)")
    print("3. Install dependencies from requirements.txt")
    print("4. Run this script with the appropriate configuration")
    print("\nThis simplified version demonstrates the project structure.")
    print("For the full implementation, refer to the generated project files.")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Basic configuration display
    config_name = args.config or "custom"
    print(f"\nConfiguration: {config_name}")
    print(f"Device: {args.device}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"WandB logging: {not args.no_wandb}")
    
    if args.data_path:
        print(f"Data path: {args.data_path}")
    elif args.download_data:
        print("Will download sample data")
    else:
        print("No data source specified. Use --data_path or --download_data")
        return
    
    print("\nTo run the complete training:")
    print("1. Implement all project modules")
    print("2. Use the provided architecture and configuration files")
    print("3. Connect to WandB for monitoring")
    print("4. Train with grandmaster chess games")


if __name__ == "__main__":
    main()