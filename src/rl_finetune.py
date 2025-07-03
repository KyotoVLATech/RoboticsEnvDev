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
    if config['debug']:
        logging.basicConfig(level=logging.DEBUG)
    else:
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
    else:
        print("Pretrained model path is not set. Using default SmolVLA configuration.")
        return
    
    # action_dimをconfigから取得、またはデフォルト値を使用
    action_dim = smolvla_policy.config.output_features["action"].shape[0]
    config['state_dim'] = smolvla_policy.config.input_features["observation.state"].shape[0]
    
    policy = SmolVLAPolicyWrapper(smolvla_policy, action_dim, config.get('initial_std', 0.1))
    trainer = PPOTrainer(env, policy, config, device)
    trainer.train()
    env.close()

if __name__ == "__main__":
    config = {
        'debug': False, # デバッグモード
        'task': 'simple_pick', # タスク名
        'observation_height': 512,
        'observation_width': 512,
        'show_viewer': False,
        'epochs': 1000, # 学習エポック数
        'smolvla_warmup_epochs': 5, # 50 SmolVLA学習開始までのエポック数
        'ppo_epochs': 4, # PPO+SmolVLAの更新エポック数
        'value_update_epochs': 10, # 価値関数の更新エポック数
        'batch_size': 1, # 1epochに実行するエピソード数
        'policy_lr': 1e-5, # log_stdとSmolVLAの学習率 SmolVLA用の学習率は2.5e-6 1e-4で勾配爆発
        'value_lr': 3e-5, # 価値関数の学習率
        'gamma': 0.99,
        'gae_lambda': 0.95, # Generalized Advantage Estimationのλパラメータ
        'clip_epsilon': 0.2,
        'kl_thresh': 100, # KLダイバージェンスの閾値
        'max_episode_steps': 300,
        'max_grad_norm': 0.5,
        'wandb_project': 'smolvla',
        'wandb_run_name': None,
        'checkpoint_dir': 'outputs/train/smolvla_ppo',
        'save_freq': 50,
        'record_video': True,
        'video_freq': 10, # 10 動画を記録する頻度
        'video_fps': 30,
        'pretrained_model_path': "outputs/train/smolvla_simple_pick/checkpoints/last/pretrained_model",
        'n_action_steps': 50, # action chunk sizeと一致させないとエラー
        'initial_std': 0.1,  # 初期標準偏差
        'entropy_coef': 0.01, # エントロピー係数
        'value_hidden_dim': 256,
        'use_image_value_network': True, # 画像特徴量を使った価値関数を使用
        'flow_matching_coef': 0.1, # Flow Matching損失の係数
    }
    main(config)
