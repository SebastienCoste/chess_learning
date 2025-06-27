from typing import Dict, Any

import torch
import torch.nn as nn

from cnn.chess.components.config import TRAINING_CONFIG


class ModelValidator:
    """
    Comprehensive model validation and verification system.
    """

    def validate_all(self, model):
        print("\n🔍 MODEL VALIDATION AND VERIFICATION")
        print("=" * 80)
        self.validate(model)
        self.test_forward_pass(model)
        print("=" * 80)

    @staticmethod
    def validate(model) -> Dict[str, Any]:
        """Validate model configuration parameters."""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }

        print(f"Model is using device {next(model.parameters()).device}")
        # Create model
        if not torch.cuda.is_available():
            validation_results['errors'].append(f"CUDA device not available but CUDA Expected.")
        if not all(param.device.type == 'cuda' for param in model.parameters()):
            validation_results['errors'].append(f"CUDA device not configured in a parameter but CUDA Expected.")
        if not all(buffer.device.type == 'cuda' for buffer in model.buffers()):
            validation_results['errors'].append(f"CUDA device not configured in a buffer but CUDA Expected.")

        # Check input channels
        if 19 != TRAINING_CONFIG["input_channels"]:
            validation_results['warnings'].append(
                f"Non-standard input channels: {TRAINING_CONFIG['input_channels']}. Chess typically uses 19 channels."
            )

        # Check board size
        if TRAINING_CONFIG['board_size'] != 8:
            validation_results['errors'].append(
                f"Invalid board size: {TRAINING_CONFIG['board_size']}. Chess board must be 8x8."
            )
            validation_results['valid'] = False

        # Check conv filters progression
        conv_filters = TRAINING_CONFIG["config"]['conv_filters']
        if not all(conv_filters[i] <= conv_filters[i + 1] for i in range(len(conv_filters) - 1)):
            validation_results['warnings'].append(
                "Convolutional filters don't follow increasing pattern."
            )

        # Check dropout rate
        if not 0.0 <= TRAINING_CONFIG["config"]['dropout_rate'] <= 0.5:
            validation_results['warnings'].append(
                f"Dropout rate {TRAINING_CONFIG["config"]['dropout_rate']} may be too high or low. Recommended: 0.1-0.5"
            )

        print("📋 Configuration Validation:")
        print("-" * 40)
        if validation_results['warnings']:
            print("⚠️  Warnings:")
            for warning in validation_results['warnings']:
                print(f"   • {warning}")
        if validation_results['recommendations']:
            print("💡 Recommendations:")
            for rec in validation_results['recommendations']:
                print(f"   • {rec}")
        if validation_results['valid']:
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration has errors:")
            for error in validation_results['errors']:
                print(f"   • {error}")
            raise Exception("❌ Configuration has errors:")

        return validation_results

    @staticmethod
    def test_forward_pass(model: nn.Module, device: str = 'cuda') -> Dict[str, Any]:
        """Test model forward pass with dummy data."""
        # Test forward pass
        print("\n🚀 Forward Pass Test:")
        print("-" * 40)
        input_shape = (2, TRAINING_CONFIG['input_channels'],
                       TRAINING_CONFIG['board_size'], TRAINING_CONFIG['board_size'])
        test_results = {
            'success': False,
            'output_shape': None,
            'error': None,
            'memory_usage': None
        }

        try:
            model.eval()
            model = model.to(device)

            # Create dummy input
            dummy_input = torch.randn(input_shape).to(device)

            # Forward pass
            with torch.no_grad():
                output = model(dummy_input)
            if output is None:
                raise Exception("Model returned None output!")

            test_results['success'] = True
            test_results['output_shape'] = tuple(output.shape)

            # Memory usage (approximate)
            test_results['device'] = device
            if device == 'cuda':
                test_results['memory_usage'] = torch.cuda.memory_allocated() / 1024 ** 2  # MB


        except Exception as e:
            test_results['error'] = str(e)

        if test_results['success']:
            print("✅ Forward pass successful")
            print(f"   • Input shape: {input_shape}")
            print(f"   • Output shape: {test_results['output_shape']}")
            if test_results['memory_usage']:
                print(f"   • Memory usage: {test_results['memory_usage']:.2f} MB")
        else:
            print("❌ Forward pass failed:")
            print(f"   • Error: {test_results['error']}")
            raise Exception("❌ Forward pass failed:")

        return test_results