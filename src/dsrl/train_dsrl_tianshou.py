"""
DSRL Training with Tianshou SAC

This script implements DSRL using tianshou's SAC algorithm with the NoiseActionEnv wrapper.
"""

import os
import sys
import logging
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import cv2

# Tianshou imports
import gymnasium as gym
import tianshou as ts
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils import TensorboardLogger, WandbLogger

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dsrl.custom_env import NoiseActionEnv, StateObsEnv, NoiseActionVisualEnv

def create_networks(state_dim: int, action_dim: int, hidden_dim: int = 256, device: str = "cuda"):
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

def create_policy(actor, critic1, critic2, config: Dict, device: str = "cuda"):
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

def _render_frame(obs: Dict, reward: float, task_desc: str = None) -> Optional[np.ndarray]:
    """動画用フレームをレンダリング"""
    frames = []
    for key in ['observation.images.front', 'observation.images.side']:
        if key in obs:
            img = obs[key]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if isinstance(img, np.ndarray):
                img = img.copy()
            # 画像の次元を調整 (C, H, W) -> (H, W, C)
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            # 値の範囲を[0,255]に正規化
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            frames.append(img)
    if frames:
        # 複数の画像を横に結合
        combined = np.concatenate(frames, axis=1)
        # 報酬を表示
        cv2.putText(
            combined, f"Reward: {reward:.2f}",
            (10, combined.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        # タスク記述を表示
        if task_desc:
            cv2.putText(
                combined,
                task_desc,
                (10, combined.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
        return combined
    return None

def _upload_video(frames: list, epoch: int) -> None:
    """動画をWandBにアップロード"""
    if not frames:
        return
    # フレームを(T, H, W, C)形式に変換
    video_array = np.stack(frames, axis=0)  # (T, H, W, C)
    # 値の範囲を[0, 255]に正規化し、uint8に変換
    if video_array.max() <= 1.0:
        video_array = (video_array * 255).astype(np.uint8)
    else:
        video_array = video_array.astype(np.uint8)
    # THWC -> TCHW
    video_array = np.transpose(video_array, (0, 3, 1, 2))
    import wandb
    wandb.log({
        f"videos/training_video": wandb.Video(video_array, fps=30, format="mp4")
    })
    logging.info(f"Uploaded training video for epoch {epoch}")

def record_training_video(env, policy, config: Dict, epoch: int) -> None:
    """学習中の動画を記録してWandBにアップロード"""
    # VectorEnvの場合は最初の環境を取得
    if hasattr(env, 'envs'):
        actual_env = env.envs[0]
        # 環境をリセット（VectorEnvの場合）
        obs = env.reset()
        if isinstance(obs, np.ndarray) and obs.ndim > 1:
            obs = obs[0]  # 最初の環境の観測を取得
    else:
        actual_env = env
        obs, info = env.reset()
    
    # タスク記述を取得
    task_desc = actual_env.get_task_description()
    frames = []
    done = False
    episode_reward = 0.0
    episode_length = 0
    max_steps = getattr(actual_env, 'max_episode_steps', 100)
    device = config.get('device', 'cuda')
    
    while not done and episode_length < max_steps:
        # フレームを記録
        if hasattr(actual_env, 'current_obs'):
            current_obs = actual_env.current_obs
        else:
            # Fallback: observationから画像を抽出
            current_obs = {}
            if isinstance(obs, dict):
                current_obs = obs
            elif hasattr(obs, '__dict__'):
                current_obs = obs.__dict__
        frame = _render_frame(current_obs, episode_reward, task_desc)
        if frame is not None:
            frames.append(frame)
        
        # アクションを選択（決定論的）
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            else:
                obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(device)
            action = policy.actor(obs_tensor)[0].cpu().numpy()
        
        # 環境をステップ実行
        if hasattr(env, 'envs'):
            # VectorEnvの場合
            obs, reward, terminated, truncated, info = env.step(np.array([action]))
            # 結果を展開
            if isinstance(obs, np.ndarray) and obs.ndim > 1:
                obs = obs[0]
            if isinstance(reward, (list, np.ndarray)):
                reward = reward[0]
            if isinstance(terminated, (list, np.ndarray)):
                terminated = terminated[0]
            if isinstance(truncated, (list, np.ndarray)):
                truncated = truncated[0]
            if isinstance(info, (list, np.ndarray)):
                info = info[0] if len(info) > 0 else {}
        else:
            obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_length += 1
        done = terminated or truncated
    
    # 最後のフレームも記録
    if hasattr(actual_env, 'current_obs'):
        current_obs = actual_env.current_obs
        frame = _render_frame(current_obs, episode_reward, task_desc)
        if frame is not None:
            frames.append(frame)
    
    # 動画をアップロード
    if frames:
        _upload_video(frames, epoch)
        logging.info(f"Recorded training video with {len(frames)} frames for epoch {epoch}")
    else:
        logging.warning(f"No frames recorded for epoch {epoch}")

def save_checkpoint(policy, epoch: int, checkpoint_dir: str):
    """チェックポイントを保存"""
    checkpoint_path = Path(checkpoint_dir) / f"epoch_{epoch}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    # Save policy
    torch.save(policy.state_dict(), checkpoint_path / "policy.pth")
    logging.info(f"Checkpoint saved to {checkpoint_path}")

def main(config: Dict):
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
    # 出力ディレクトリの作成
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    # Environment作成 - Genesis制限により単一環境インスタンスを使用
    train_envs = make_env(config)
    action_space = train_envs.action_space
    state_dim = train_envs.observation_space.shape[0]
    action_dim = action_space.shape[0]
    train_envs = DummyVectorEnv([lambda : train_envs])
    logging.info(f"Environment: state_dim={state_dim}, action_dim={action_dim}")
    # Networks and Policy
    actor, critic1, critic2 = create_networks(
        state_dim, action_dim, config.get('hidden_dim', 256), device
    )
    policy = create_policy(actor, critic1, critic2, config, device)
    # Replay buffer
    buffer = VectorReplayBuffer(
        config.get('buffer_size', 100000),
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
        try:
            # wandb.initは既に行われているので、引数なしでWandbLoggerを初期化
            logger = WandbLogger(test_interval=config.get('log_per_epoch', 1))
            logger.load(writer)
            logging.info("WandB logger initialized successfully")
        except Exception as e:
            logging.warning(f"Failed to initialize WandB logger: {e}")
            logging.info("Falling back to TensorBoard logger")
            logger = TensorboardLogger(writer)
    else:
        logger = TensorboardLogger(writer)

    # Checkpoint function with video recording
    def save_fn(epoch, env_step, gradient_step):
        save_checkpoint(policy, epoch, config['checkpoint_dir'])
        # 動画記録とアップロード
        if (config.get('record_video', False) and config['task'] != 'pendulum' and train_envs is not None):
            record_training_video(train_envs, policy, config, epoch)

    # Training - Genesis環境での制限を考慮
    result = offpolicy_trainer(
        policy,
        train_collector,
        None,
        max_epoch=config.get('max_epoch', 100),
        step_per_epoch=config.get('step_per_epoch', 10000),
        step_per_collect=config.get('step_per_collect', 10),
        episode_per_test=0,
        batch_size=config.get('batch_size', 256),
        update_per_step=config.get('update_per_step', 0.1),
        test_in_train=False,
        logger=logger,
        save_checkpoint_fn=save_fn,
        resume_from_log=config.get('resume_from_log', False),
        verbose=True
    )

    train_envs.close()
    logging.info("Training completed!")
    return result

def evaluate_policy(checkpoint_path: str, config: Dict, num_episodes: int = 10):
    """学習済みポリシーの評価"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config['device'] = device
    
    # Pendulum環境の場合はshow_viewerを設定しない
    if config['task'] != 'pendulum':
        config['show_viewer'] = True  # 評価時は表示
    
    # Environment
    env = make_env(config)
    
    # Networks and Policy
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    actor, critic1, critic2 = create_networks(
        state_dim, action_dim, config.get('hidden_dim', 256), device
    )
    policy = create_policy(actor, critic1, critic2, config, device)
    
    # Load checkpoint
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()
    
    # Evaluation
    total_rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        # Pendulum環境の場合はreset()の返り値が異なる
        if config['task'] == 'pendulum':
            obs, info = env.reset(), {}
        else:
            obs, info = env.reset()
        
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        # Video recording for non-pendulum environments
        if config['task'] != 'pendulum' and hasattr(env, 'start_video_recording'):
            env.start_video_recording()
        
        # Set max episode steps
        if config['task'] == 'pendulum':
            max_steps = 200  # Pendulum standard episode length
        else:
            max_steps = env.max_episode_steps
        
        while not done and episode_length < max_steps:
            # Deterministic action selection
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = policy.actor(obs_tensor)[0].cpu().numpy()
            
            # Step environment
            if config['task'] == 'pendulum':
                obs, reward, terminated, truncated, info = env.step(action)
            else:
                obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        # Stop video recording for non-pendulum environments
        if config['task'] != 'pendulum' and hasattr(env, 'stop_video_recording'):
            frames = env.stop_video_recording()
        
        total_rewards.append(episode_reward)
        if info.get('is_success', False):
            success_count += 1
        
        logging.info(f"Episode {episode + 1}: reward={episode_reward:.3f}, "
                    f"length={episode_length}, success={info.get('is_success', False)}")
    
    # Results
    avg_reward = np.mean(total_rewards)
    success_rate = success_count / num_episodes
    
    logging.info(f"\nEvaluation Results ({num_episodes} episodes):")
    logging.info(f"Average Reward: {avg_reward:.3f}")
    logging.info(f"Success Rate: {success_rate:.3f} ({success_count}/{num_episodes})")
    
    env.close()
    return avg_reward, success_rate

if __name__ == "__main__":
    # Configuration
    config = {
        # Environment settings
        'task': 'simple_pick', # pendulum, simple_pick, vla_pick, vla_visual_pick
        'observation_height': 512,
        'observation_width': 512,
        'show_viewer': False,

        'max_epoch': 500,
        'step_per_epoch': 10, # 1エポックあたりの学習ステップ数
        'step_per_collect': 100, # 1回の学習ステップで収集する環境のステップ数
        'batch_size': 4,
        'update_per_step': 1, # update_per_step: the number of times the policy network would be updated per transition after (step_per_collect) transitions are collected, e.g., if update_per_step set to 0.3, and step_per_collect is 256 , policy will be updated round(256 * 0.3 = 76.8) = 77 times after 256 transitions are collected by the collector. Default to 1.

        # Network settings
        'hidden_dim': 256,
        'learning_rate': 3e-4,
        'buffer_size': 100000,

        # SAC hyperparameters
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.2,
        'estimation_step': 1,

        # DSRL settings
        'chunk_size': 50,

        # Logging and saving
        'use_wandb': True,
        'wandb_project': 'smolvla',
        'wandb_run_name': None,
        'checkpoint_dir': 'outputs/train/dsrl_tianshou_checkpoints',
        'resume_from_log': False,
        'log_per_epoch': 1,

        # Video recording settings
        'record_video': True,  # Enable video recording during training

        # SmolVLA settings
        'pretrained_model_path': 'outputs/train/smolvla_simple_pick/checkpoints/last/pretrained_model',
        'smolvla_config_overrides': {
            'n_action_steps': 10,
        }
    }

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    eval = False
    if eval:
        # set eval parameters
        checkpoint_path = 'hogehoge'
        num_episodes = 10
        evaluate_policy(checkpoint_path, config, num_episodes)
    else:
        main(config)
