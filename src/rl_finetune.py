"""PPO training for SmolVLA policy."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import logging
from pathlib import Path
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.genesis_env import GenesisEnv
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from src.rl_agent import SmolVLAPolicyWrapper, PPOTrainer


def main(config):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    env = GenesisEnv(
        task=config['task'],
        observation_height=config['observation_height'],
        observation_width=config['observation_width'],
        show_viewer=config['show_viewer']
    )
    
    if config['pretrained_model_path']:
        smolvla_policy = SmolVLAPolicy.from_pretrained(config['pretrained_model_path'])
        smolvla_policy.config.n_action_steps = config['n_action_steps']
    else:
        smolvla_config = SmolVLAConfig()
        smolvla_config.n_action_steps = config['n_action_steps']
        smolvla_policy = SmolVLAPolicy(smolvla_config)
    
    # action_dimをconfigから取得、またはデフォルト値を使用
    action_dim = smolvla_policy.config.output_features.action.shape[0]
    config['state_dim'] = smolvla_policy.config.input_features.state.shape[0]
    
    policy = SmolVLAPolicyWrapper(smolvla_policy, action_dim, config.get('initial_std', 0.1))
    trainer = PPOTrainer(env, policy, config, device)
    trainer.train()
    env.close()

if __name__ == "__main__":
    config = {
        'task': 'simple_pick',
        'observation_height': 512,
        'observation_width': 512,
        'show_viewer': False,
        'epochs': 1000,
        'batch_size': 8,  # 並行して実行するエピソード数
        'policy_lr': 1e-5,
        'value_lr': 1e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'ppo_epochs': 4,
        'max_episode_steps': 500,
        'max_grad_norm': 0.5,
        'wandb_project': 'smolvla-ppo',
        'wandb_run_name': None,
        'checkpoint_dir': 'outputs/rl_checkpoints_ppo',
        'save_freq': 50,
        'record_video': True,
        'video_freq': 10,
        'video_fps': 30,
        'pretrained_model_path': "outputs/train/smolvla_test_0/checkpoints/last/pretrained_model",
        'n_action_steps': 20,
        'initial_std': 0.1,  # 初期標準偏差
        'entropy_coef': 0.01,  # エントロピー係数
        'target_kl': 0.02,  # KLダイバージェンスの閾値
        'value_hidden_dim': 256,
        'value_stable_threshold': 0.01,  # 価値関数の安定性判定閾値
        'value_stable_window': 10,       # 安定性判定のためのウィンドウサイズ
        'value_update_epochs': 5,        # 価値関数の更新エポック数
        
        # ハイブリッド学習用の新しいパラメータ
        'smolvla_lr': 1e-6,              # SmolVLA用の学習率（PPOより低く設定）
        'smolvla_warmup_epochs': 50,     # SmolVLA学習開始までのエポック数
        'flow_matching_coef': 0.1,       # Flow Matching損失の係数
    }
    main(config)
