INPUT_CHANNEL = 19

TRAINING_CONFIG = {
    "num_epoch": 200,
    "device": "cuda",
    "kernel_size": 3,
    "spatial_kernel_size": 3,
    "input_channels": INPUT_CHANNEL,
    "board_size": 8,
    "batch_size": 2048,  # To be adjusted. 128 is way too small5
    "num_workers": 10, #8 cores, 16 virtual workers
    "version": "v5.2",
    "learning_rate": 0.0003,
    "weight_decay": 1e-3, #increased from 1e-4 to fight overfitting
    "scheduler_type": 'cosine_annealing_warm_restarts', #'cosine_annealing', #'reduce_on_plateau'
    "early_stopping_patience": 5,
    "mixed_precision": True,
    "with_ema": False,
    "with_mixup": False,
    "gradient_clipping": 0.5, #Reduced from 1.0 because of gradient instability
    'accumulation_steps': 4,
    "pth_file": "chess_gm_puzzle",
    "cache_type": "lru", # lru or shared
    "config": {
            'input_channels': INPUT_CHANNEL,
            'board_size': 8,
            'conv_filters': [64, 128, 256],
            'fc_layers': [512, 256],
            'dropout_rate': 0.5, #increased from 0.3
            'batch_norm': True,
            'activation': 'relu',
            'transformer_heads': 4,
        },
    "cosine": {
        "warmup_epochs": 3, #or 10 ? # Ramp up learning rate gradually
        "first_restart": 5, #or 10 ? # Ramp up learning rate gradually
        "min_lr": 1e-6,
        "eta_min": 1e-6 ,     # Minimum LR for cosine annealing
    }
}
'''
Baseline: 
Batch of 1024
num workers: t:2, v: 1
cache OptimizedMemmapChessDataset (other)
activation: mish

Throughput samples/sec at 500k samples: 
Baseline            : 2000 (batch 500)
Batch -> 2048       : 2000 (batch 250)
num worker -> t:24  : 3600 (SSD working a lot more, so maybe ram loaded faster?)
cache -> lru        : 3700 (no cache hit rate, some better spikes of GPU)
num worker -> t:12  : 3830
num worker -> t:6   : 3770 
num worker -> t:10  : 3760 (going back to 6 the variation is not significant)
activate -> relu    : 3900

'''