# DSRL Implementation with Tianshou

This directory contains the DSRL (Diffusion Steering via Reinforcement Learning) implementation that has been migrated to use the tianshou library.

## File Structure

### Core Files

- **`dsrl.py`**: Contains the `SmolVLAWrapper` class and `load_smolvla_model` function. This is the core DSRL component that handles SmolVLA model integration.

- **`noise_action_env.py`**: Contains the `NoiseActionEnv` class, a gym-compatible wrapper that:
  - Takes latent noise as actions from RL agent
  - Uses SmolVLAPolicy to convert noise to actual actions
  - Executes actual actions in GenesisEnv
  - Returns rich state features as observations

- **`train_dsrl_tianshou.py`**: Main training script using tianshou's SAC implementation. Includes:
  - Network creation functions
  - Training loop using tianshou's `offpolicy_trainer`
  - Evaluation functions
  - WandB integration

### Legacy Files (Not Used)

- **`dsrl_agent.py`**: Contains old custom SAC and NA implementations (not used anymore)
- **`rl_agent.py`**: Old RL agent implementations (not used anymore)
- **`rl_finetune.py`**: Old fine-tuning script (not used anymore)

## Usage

### Training

To train a DSRL agent using tianshou SAC:

```bash
cd src
uv run src/train_dsrl_tianshou.py
```

For algorithm testing with Pendulum environment:
```bash
uv run src/train_dsrl_tianshou.py --task pendulum --no-wandb
```

For DSRL with SmolVLA:
```bash
uv run src/train_dsrl_tianshou.py \
    --task simple_pick \
    --pretrained_model_path outputs/train/smolvla_simple_pick/checkpoints/last/pretrained_model
```

### Evaluation

To evaluate a trained model:

```bash
uv run src/train_dsrl_tianshou.py \
    --eval \
    --checkpoint_path outputs/train/dsrl_tianshou_checkpoints/epoch_100/policy.pth \
    --num_episodes 10
```

## Key Changes from Original Implementation

1. **Replaced Custom SAC/NA**: Removed custom DSRL-SAC and DSRL-NA implementations in favor of tianshou's optimized SAC.

2. **Environment Wrapper**: Created `NoiseActionEnv` that cleanly separates the noise→action conversion from the training loop.

3. **Simplified Architecture**: 
   - `SmolVLAWrapper`: Handles SmolVLA integration and feature extraction
   - `NoiseActionEnv`: Gym-compatible environment wrapper
   - `train_dsrl_tianshou.py`: Training script using tianshou

4. **Better Integration**: Leverages tianshou's:
   - Optimized SAC implementation
   - VectorReplayBuffer
   - Parallel environment support
   - Logging and checkpointing utilities

## Configuration

The training script uses a configuration dictionary that can be modified in the script or overridden via command line arguments. Key parameters:

```python
config = {
    # Environment
    'task': 'simple_pick',
    'observation_height': 512,
    'observation_width': 512,
    
    # Training
    'max_epoch': 200,
    'step_per_epoch': 5000,
    'batch_size': 256,
    
    # SAC hyperparameters
    'gamma': 0.99,
    'tau': 0.005,
    'alpha': 0.2,
    
    # DSRL
    'chunk_size': 50,
    
    # Logging
    'use_wandb': True,
    'wandb_project': 'dsrl-smolvla-tianshou',
}
```

## Notes

- The implementation maintains the core DSRL concept of learning in noise space
- SmolVLA model remains frozen (black-box)
- Action chunking is handled automatically
- WandB logging includes video recording of evaluation episodes
- Checkpoints are saved in tianshou format
