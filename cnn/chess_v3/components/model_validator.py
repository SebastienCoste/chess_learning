import torch
import torch.nn as nn
from typing import Dict, Tuple, Any

INPUT_CHANNELS = 20

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