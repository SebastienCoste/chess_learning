# Limit autotuning to reduce memory-intensive configurations
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="ATEN,TRITON"
export TORCHINDUCTOR_TRITON_AUTOTUNE_MAX_BLOCKS=8192

# Force smaller block sizes during autotuning
export TORCHINDUCTOR_AUTOTUNE_MAX_CONFIGS=5

# Disable caching to force recompilation with new settings
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

python ./train_model.py