INPUT_CHANNEL = 19

TRAINING_CONFIG = {
    "num_epoch": 200,
    "device": "cuda",
    "kernel_size": 3,
    "input_channels": INPUT_CHANNEL,
    "board_size": 8,
    "batch_size": 512,  # To be adjusted. 128 is way too small5
    "num_workers": 16, #8 cores, 16 virtual workers
    "version": "v2",
    "learning_rate": 0.001,
    "weight_decay": 1e-3, #increased from 1e-4 to fight overfitting
    "scheduler_type": 'cosine', #'reduce_on_plateau'
    "early_stopping_patience": 10,
    "mixed_precision": True,
    "config": {
            'input_channels': INPUT_CHANNEL,
            'board_size': 8,
            'conv_filters': [64, 128, 256],
            'fc_layers': [512, 256],
            'dropout_rate': 0.4, #increased from 0.3
            'batch_norm': True,
            'activation': 'gelu',
            'use_attention': True,
            'use_transformer_blocks': True,
            'num_transformer_layers': 2,
            'transformer_heads': 8,
        },
    "cosine": {
        "warmup_epochs": 10,  # Ramp up learning rate gradually
        "min_lr": 1e-7,
        "eta_min": 1e-7 ,     # Minimum LR for cosine annealing
    }
}
