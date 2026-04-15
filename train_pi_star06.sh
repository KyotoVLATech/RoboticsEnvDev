#!/bin/bash

export DATASET_NAME=normal_0

uv run -m lerobot.rl.algorithms.recap_train_pi_star \
    --repo_id=local/${DATASET_NAME} \
    --root=datasets/${DATASET_NAME} \
    --output_dir="outputs/recap_pistar" \
    --value_network_checkpoint="outputs/recap_value_0/checkpoints/last.pt" \
    --epochs=5 \
    --batch_size=6 \
    --learning_rate=1e-4 \
    --val_split_ratio=0.1 \
    --validate_every_n_train_steps=50 \
    --advantage_threshold=0.0 \
    --advantage_dropout=0.3 \
    --log_every_n_steps=10 \
    --model_precision="bfloat16" \
    --freeze_vision_encoder=true \
    --advantage_cache_path="outputs/advantage_cache.json"