# Install dependencies
pip install -r requirements.txt  # Includes wandb, torch, python-chess, optuna

# Start training with optimal parameters
python train.py \
  --conv_layers 5 \
  --init_filters 96 \
  --dense_units 384 \
  --batch_size 1024 \
  --lr 0.001 \
  --wandb

# Launch hyperparameter search
python hpo.py \
  --n_trials 100 \
  --pruner hyperband \
  --direction maximize

# Start interactive play
python play.py --model best_model.pth
