# hpo.py
import optuna


def objective(trial):
    params = {
        'conv_layers': trial.suggest_int('conv_layers', 3, 6),
        'init_filters': trial.suggest_categorical('init_filters', [32, 64, 128]),
        'dense_units': trial.suggest_int('dense_units', 128, 512, step=64),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    }

    model = ChessCNN(**params)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=params['lr'],
                                  weight_decay=params['weight_decay'])

    # Training loop with wandb integration
    wandb.init(config=params)
    train_model(model, optimizer, params['batch_size'])
    return evaluate_model(model)
