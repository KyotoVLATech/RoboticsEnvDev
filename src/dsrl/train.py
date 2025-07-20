import os
import sys
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict
import gymnasium as gym
from tianshou.policy import SACPolicy, PPOPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils import TensorboardLogger, WandbLogger
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.dsrl.custom_env import NoiseActionEnv, StateObsEnv, NoiseActionVisualEnv, BaseCustomEnv
from src.dsrl.custom_trainer import dsrl_trainer

def create_sac_networks(state_dim: int, action_dim: int, hidden_dim: int = 256, device: str = "cuda"):
    """SAC用のアクター・クリティックネットワークを作成"""
    
    # Actor network
    net_a = Net(state_dim, hidden_sizes=[hidden_dim, hidden_dim], device=device)
    actor = ActorProb(
        net_a, action_dim, max_action=1.0, device=device,
        unbounded=True, conditioned_sigma=True
    ).to(device)
    
    # Critic networks
    net_c1 = Net(
        state_dim, action_dim, hidden_sizes=[hidden_dim, hidden_dim], 
        concat=True, device=device
    )
    critic1 = Critic(net_c1, device=device).to(device)
    
    net_c2 = Net(
        state_dim, action_dim, hidden_sizes=[hidden_dim, hidden_dim],
        concat=True, device=device
    )
    critic2 = Critic(net_c2, device=device).to(device)
    return actor, critic1, critic2

def create_ppo_networks(state_dim: int, action_dim: int, hidden_dim: int = 256, device: str = "cuda"):
    """PPO用のアクター・クリティックネットワークを作成"""
    # Actor network
    net_a = Net(state_dim, hidden_sizes=[hidden_dim, hidden_dim], device=device)
    actor = ActorProb(
        net_a, action_dim, max_action=1.0, device=device,
        unbounded=True, conditioned_sigma=True
    ).to(device)
    # Critic network
    net_c = Net(state_dim, hidden_sizes=[hidden_dim, hidden_dim], device=device)
    critic = Critic(net_c, device=device).to(device)
    return actor, critic

def create_sac_policy(actor, critic1, critic2, config: Dict, device: str = "cuda"):
    """SAC Policyを作成"""
    # Optimizers
    optim_actor = torch.optim.Adam(actor.parameters(), lr=config.get('learning_rate', 3e-4))
    optim_critic1 = torch.optim.Adam(critic1.parameters(), lr=config.get('learning_rate', 3e-4))
    optim_critic2 = torch.optim.Adam(critic2.parameters(), lr=config.get('learning_rate', 3e-4))
    # SAC Policy
    policy = SACPolicy(
        actor=actor,
        critic1=critic1, 
        critic2=critic2,
        actor_optim=optim_actor, 
        critic1_optim=optim_critic1, 
        critic2_optim=optim_critic2,
        tau=config.get('tau', 0.005),
        gamma=config.get('gamma', 0.99),
        alpha=config.get('alpha', 0.2),
        estimation_step=config.get('estimation_step', 1),
        action_space=None  # Will be set by the environment
    )
    
    return policy

def create_ppo_policy(actor, critic, config: Dict, device: str = "cuda"):
    """PPO Policyを作成"""
    
    # Optimizers
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), 
        lr=config.get('learning_rate', 3e-4)
    )
    
    # Value function
    def dist_fn(*logits):
        # logitsは展開されて渡される場合があるので適切に処理
        if len(logits) == 2:
            # 平均と標準偏差が別々に渡される場合
            mean, std = logits
        elif len(logits) == 1:
            # 単一のテンソルの場合、平均と標準偏差を分割
            logits = logits[0]
            if isinstance(logits, tuple):
                mean, std = logits
            else:
                mean, std = logits.chunk(2, dim=-1)
                std = torch.clamp(std, min=-20, max=2).exp()
        else:
            raise ValueError(f"Unexpected number of logits: {len(logits)}")
        return torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)
    
    # PPO Policy
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,
        discount_factor=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        vf_coef=config.get('vf_coef', 0.5),
        ent_coef=config.get('ent_coef', 0.01),
        eps_clip=config.get('eps_clip', 0.2),
        value_clip=config.get('value_clip', True),
        dual_clip=config.get('dual_clip', None),
        advantage_normalization=config.get('advantage_normalization', True),
        recompute_advantage=config.get('recompute_advantage', False),
        action_space=None  # Will be set by the environment
    )
    return policy

def create_policy(algorithm: str, state_dim: int, action_dim: int, config: Dict, device: str = "cuda"):
    """アルゴリズムに応じたpolicyを作成"""
    algorithm = algorithm.lower()
    hidden_dim = config.get('hidden_dim', 256)
    if algorithm == 'sac':
        actor, critic1, critic2 = create_sac_networks(state_dim, action_dim, hidden_dim, device)
        return create_sac_policy(actor, critic1, critic2, config, device)
    elif algorithm == 'ppo':
        actor, critic = create_ppo_networks(state_dim, action_dim, hidden_dim, device)
        return create_ppo_policy(actor, critic, config, device)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

def make_env(config: Dict):
    """環境を作成する関数"""
    task = config['task']
    if task == 'pendulum':
        env = gym.make('Pendulum-v1')
        return env
    elif task == 'simple_pick':
        env = StateObsEnv(config)
        return env
    elif task == 'vla_pick':
        env = NoiseActionEnv(config)
        return env
    elif task == 'vla_visual_pick':
        env = NoiseActionVisualEnv(config)
        return env
    else:
        raise ValueError(f"Unknown task: {task}")

def record_training_video(env, policy, config: Dict, epoch: int) -> None:
    """学習中の動画を記録してWandBにアップロード"""
    actual_env : BaseCustomEnv = env.workers[0].env
    # 環境をリセット（VectorEnvの場合）
    obs = env.reset()
    frames = []
    done = False
    episode_length = 0
    max_steps = getattr(actual_env, 'max_episode_steps', 100)
    device = config.get('device', 'cuda')

    while not done and episode_length < max_steps:
        frame = actual_env.render_frame_with_info()
        if frame is not None:
            frames.append(frame)
        # アクションを選択（決定論的）
        obs_array = obs[0]
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(device)
            # PPOのactorの出力: ((mean, std), None)
            actor_output = policy.actor(obs_tensor)
            # 最初の要素から平均を取得
            mean_std_tuple = actor_output[0]
            mean_tensor = mean_std_tuple[0]  # 平均テンソル
            # 決定論的アクションとして平均を使用
            action = mean_tensor.cpu().numpy()[0]  # バッチ次元を削除
        # 環境をステップ実行
        obs, reward, terminated, truncated, info = env.step(np.array([action]))
        episode_length += 1
        done = terminated[0] or truncated[0]
    # 動画をアップロード
    actual_env.upload_video_to_wandb(frames, epoch)

def save_checkpoint(policy, epoch: int, checkpoint_dir: str):
    """チェックポイントを保存し、ファイルパスを返す"""
    checkpoint_path = Path(checkpoint_dir) / f"epoch_{epoch}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    # Save policy
    save_path = checkpoint_path / "policy.pth"
    torch.save(policy.state_dict(), save_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")
    return str(save_path)

def main(config: Dict):
    if config['task'] == 'pendulum':
        config['record_video'] = False  # Pendulum does not support video recording
    if config.get('use_wandb', True):
        import wandb
        wandb.init(
            project=config.get('wandb_project', 'smolvla'),
            name=config.get('wandb_run_name'),
            config=config,
            sync_tensorboard=True
        )
    # Device設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config['device'] = device
    logging.info(f"Using device: {device}")
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    train_envs = make_env(config)
    action_space = train_envs.action_space
    state_dim = train_envs.observation_space.shape[0]
    action_dim = action_space.shape[0]
    train_envs = DummyVectorEnv([lambda : train_envs])
    logging.info(f"Environment: state_dim={state_dim}, action_dim={action_dim}")
    # Policy creation
    policy = create_policy(config['algorithm'], state_dim, action_dim, config, device)
    # Replay buffer
    if config['algorithm'].lower() in ['sac', 'ddpg', 'td3']:
        buffer = VectorReplayBuffer(
            config.get('buffer_size', 100000),
            config.get('train_num', 1)
        )
    else:
        # On-policy algorithms don't need a replay buffer in the same way
        buffer = VectorReplayBuffer(
            config.get('step_per_collect', 2048),
            config.get('train_num', 1)
        )
    # Collectors
    train_collector = Collector(
        policy, train_envs, buffer, exploration_noise=True
    )
    # Logger
    from torch.utils.tensorboard import SummaryWriter
    log_path = Path(config['checkpoint_dir']) / 'logs'
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_path))
    if config.get('use_wandb', True):
        # wandb.initは既に行われているので、引数なしでWandbLoggerを初期化
        logger = WandbLogger(test_interval=config.get('log_per_epoch', 1))
        logger.load(writer)
    else:
        logger = TensorboardLogger(writer)
    config['logger'] = logger

    # Checkpoint function with configurable save interval
    def save_fn(epoch, env_step, gradient_step):
        save_interval = config.get('save_checkpoint_interval', 1)  # Default: save every epoch
        if epoch % save_interval == 0:
            path = save_checkpoint(policy, epoch, config['checkpoint_dir'])
            logging.info(f"Checkpoint saved at epoch {epoch} (interval: {save_interval})")
            return path
        # Noneの代わりにダミーファイルを返す
        dummy_path = Path(config['checkpoint_dir']) / "dummy.pth"
        if not dummy_path.exists():
            dummy_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_path.touch()
        return str(dummy_path)

    config['save_checkpoint_fn'] = save_fn
    # Training with unified trainer
    result = dsrl_trainer(
        algorithm=config['algorithm'],
        policy=policy,
        train_collector=train_collector,
        test_collector=None,
        max_epoch=config.get('max_epoch', 100),
        step_per_epoch=config.get('step_per_epoch', 10000),
        batch_size=config.get('batch_size', 256),
        config=config,
        record_video_fn=record_training_video if (config.get('record_video', False) and config.get('use_wandb', True)) else None,
        train_envs=train_envs if config.get('record_video', False) else None,
    )

    train_envs.close()
    logging.info("Training completed!")
    return result

if __name__ == "__main__":
    # pendulum: アルゴリズム検証用
    # simple_pick: SmolVLAを使わない普通のSimplePickタスク．joint位置，速度，目標とエンドエフェクタの相対座標をobservationとする
    # vla_pick: SmolVLAを使ったSimplePickタスク．observationはsimple_pickと同じ
    # vla_visual_pick: SmolVLAを使ったSimplePickタスク．SmolVLAのエンコーダから取得した特徴量をobservationとする．（テキスト，画像，自己受容状態）
    task = 'vla_pick'
    algorithm = 'ppo'  # 使用する強化学習アルゴリズム（'sac' または 'ppo'）

    # Configuration
    config = {
        # Algorithm settings
        'algorithm': algorithm,

        # Environment settings
        'task': task,
        'observation_height': 512, # 観測画像の高さ（基本的に変更不要）
        'observation_width': 512,  # 観測画像の幅（基本的に変更不要）
        'show_viewer': False, # TrueでGenesisのViewerを表示

        # Training settings
        'max_epoch': 1000,  # 学習エポック数
        'step_per_epoch': 1000, # 4096 1エポックあたりの学習ステップ数 PPOならstep_per_collectと同じにする
        'step_per_collect': 1000, # 1回の収集で環境から集めるデータのステップ数
        'batch_size': 64,  # 256 バッチサイズ
        'update_per_step': 1, # 1ステップごとのネットワーク更新回数（Off-policy用）
        'repeat_per_collect': 10,  # 1回の収集ごとのネットワーク更新回数（On-policy用）

        # Network settings
        'hidden_dim': 256,  # ニューラルネットワークの隠れ層の次元数
        'learning_rate': 3e-4,  # 学習率
        'buffer_size': 100000,  # リプレイバッファのサイズ

        # Algorithm-specific hyperparameters
        # SAC
        'gamma': 0.99,  # 割引率（SAC/PPO共通）
        'tau': 0.005,  # ターゲットネットワークのソフト更新率（SAC用）
        'alpha': 0.2,  # エントロピー正則化係数（SAC用）
        'estimation_step': 1,  # 状態価値推定のステップ数（SAC用）TD-N

        # PPO
        'gae_lambda': 0.95,  # GAEのλパラメータ（PPO用）
        'max_grad_norm': 0.5,  # 勾配クリッピングの最大ノルム（PPO用）
        'vf_coef': 0.5,  # 価値関数損失の重み（PPO用）
        'ent_coef': 0.01,  # エントロピー損失の重み（PPO用）
        'eps_clip': 0.2,  # クリッピング範囲（PPO用）
        'value_clip': True,  # 価値関数のクリッピング有無（PPO用）
        'advantage_normalization': True,  # アドバンテージ正規化の有無（PPO用）

        # Logging and saving
        'use_wandb': True,  # WandBによるロギングを有効化
        'wandb_project': 'smolvla', # プロジェクト名
        'wandb_run_name': None, # Noneなら自動生成
        'checkpoint_dir': f'outputs/train/dsrl_{algorithm}_{task}_0',  # チェックポイント保存ディレクトリ
        'resume_from_log': False,  # ログから学習を再開するか（Trueの動作未確認
        'log_per_epoch': 1, # 何エポックごとにログを記録するか
        'save_checkpoint_interval': 100, # 何エポックごとにチェックポイントを保存するか

        # Video recording settings
        'record_video': True,  # 学習中の動画記録を有効化（pendulumなら自動で無効化）
        'video_record_interval': 50,  # 何エポックごとに動画を記録するか

        # SmolVLA settings
        'pretrained_model_path': 'outputs/train/smolvla_simple_pick/checkpoints/last/pretrained_model',  # SmolVLAの事前学習モデルパス
        'smolvla_config_overrides': {  # SmolVLAの設定上書き
            'n_action_steps': 10,  # 生成したaction chunkの内，何ステップを使用するか
        }
    }

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    main(config)