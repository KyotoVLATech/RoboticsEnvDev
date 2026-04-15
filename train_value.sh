#!/bin/bash

export DATASET_NAME=normal_0

uv run -m lerobot.rl.algorithms.recap_train_value_network \
    --repo_id="jackvial/so101_pickplace_recap_merged_v2" \
    --output_dir="outputs/recap_value_0" \
    --epochs=2 \
    --batch_size=4 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-4 \
    --num_workers=4 \
    --val_split_ratio=0.1 \
    --log_every_n_steps=100 \
    --validate_every_n_train_steps=50 \
    --plot_every_n_train_steps=200 \
    --max_val_steps_per_step_validation=20 \
    --c_fail=500.0 \
    --num_value_bins=50 \
    --num_vlm_layers=10 \
    --paligemma_variant=gemma_2b \
    --pretrained_path=lerobot/pi05_base \
    --val_plot_num_episodes=4 \
    --val_plot_num_frames=8 \
    --val_plot_every_n_epochs=1 \
    --model_precision="bfloat16" \
    --freeze_vision_encoder=true \
    --freeze_backbone=true \
    --num_unfrozen_backbone_layers=3

# uv run -m lerobot.rl.algorithms.recap_train_value_network \
#     --repo_id=local/${DATASET_NAME} \
#     --root=datasets/${DATASET_NAME} \
#     --output_dir="outputs/recap_value_0" \
#     --epochs=2 \
#     --batch_size=4 \
#     --gradient_accumulation_steps=4 \
#     --learning_rate=1e-4 \
#     --num_workers=4 \
#     --val_split_ratio=0.1 \
#     --log_every_n_steps=100 \
#     --validate_every_n_train_steps=50 \
#     --plot_every_n_train_steps=200 \
#     --max_val_steps_per_step_validation=20 \
#     --c_fail=500.0 \
#     --num_value_bins=50 \
#     --num_vlm_layers=10 \
#     --paligemma_variant=gemma_2b \
#     --pretrained_path=lerobot/pi05_base \
#     --val_plot_num_episodes=4 \
#     --val_plot_num_frames=8 \
#     --val_plot_every_n_epochs=1 \
#     --model_precision="bfloat16" \
#     --freeze_vision_encoder=true \
#     --freeze_backbone=true \
#     --num_unfrozen_backbone_layers=3