# config.py
FAST_TRAIN = {
    "batch_size": 1024,  # Maximize GPU utilization
    "mixed_precision": True,  # FP16 training
    "gradient_accumulation": 2,
    "optimizer": "Lamb",  # Converges faster than Adam
    "lr": 3e-3,
    "lr_schedule": "onecycle",  # Super convergence
    "max_epochs": 50,
    "early_stopping": 5,
    "augmentation": "light"  # Random flips/rotations
}

# Recommended for initial experiments
DEFAULT_PARAMS = {
    "conv_layers": 5,
    "init_filters": 96,
    "dense_units": 384,
    "dropout": 0.25,
    "activation": "swish"
}
