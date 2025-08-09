import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import logging
from pathlib import Path
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.genesis_env import GenesisEnv
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
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
    
    policy = SmolVLAPolicyWrapper(smolvla_policy, action_dim, config.get('initial_std', 0.1), config.get('min_std', 0.05))
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
        'epochs': 1500, # 学習エポック数
        'smolvla_warmup_epochs': 500, # SmolVLA学習開始までのエポック数 50
        'ppo_epochs': 4, # PPO+SmolVLAの更新エポック数
        'value_update_epochs': 10, # 価値関数の更新エポック数
        'batch_size': 1, # 1epochに実行するエピソード数
        'log_std_lr': 1e-2, # log_stdの学習率
        'smolvla_lr': 1e-5, # SmolVLAの学習率 LeRobotのSmolVLAのデフォルトの学習率は2.5e-6 1e-6は小さすぎる気がする
        'value_lr': 1e-4, # 価値関数の学習率 3e-5
        'gamma': 0.99,
        'gae_lambda': 0.95, # Generalized Advantage Estimationのλパラメータ
        'clip_epsilon': 0.2,
        'kl_thresh': 1.0, # KLダイバージェンスの閾値
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
        'initial_std': 1.0,  # 初期標準偏差
        'min_std': 0.05, # 最小標準偏差
        'entropy_coef': 0.001, # エントロピー係数 これが小さすぎると，log_stdが小さくなり，ratioが大きくなり，勾配爆発が起きる．0.01は小さい 0.02は大きい std_meanが0.4前後で安定する値を見つける．
        'value_hidden_dim': 256,
        'use_image_value_network': True, # 画像特徴量を使った価値関数を使用
        'flow_matching_coef': 0.0, # Flow Matching損失の係数
    }
    main(config)