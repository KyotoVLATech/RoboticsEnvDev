#!/bin/bash

export DATASET_NAME=normal-fix_1
export POLICY=diffusion

uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --dataset.video_backend=pyav --batch_size=32 --steps=100000 --policy.use_separate_rgb_encoder_per_camera=true
