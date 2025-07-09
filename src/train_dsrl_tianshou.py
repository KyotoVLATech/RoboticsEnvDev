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

from env.genesis_env import GenesisEnv
from src.dsrl import SmolVLAWrapper, load_smolvla_model
from src.noise_action_env import NoiseActionEnv

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
        # Pendulum環境での動作確認用
        env = gym.make('Pendulum-v1')
        return env
    else:
        # GenesisEnv + SmolVLA環境
        genesis_env = GenesisEnv(
            task=config['task'],
            observation_height=config['observation_height'],
            observation_width=config['observation_width'],
            show_viewer=config.get('show_viewer', False)
        )
        
        # SmolVLA model
        smolvla_policy = load_smolvla_model(
            config['pretrained_model_path'],
            config['smolvla_config_overrides']
        )
        
        # SmolVLA wrapper
        smolvla_wrapper = SmolVLAWrapper(smolvla_policy, config['device'])
        
        # NoiseActionEnv wrapper
        env = NoiseActionEnv(
            genesis_env=genesis_env,
            smolvla_wrapper=smolvla_wrapper,
            chunk_size=config.get('chunk_size', 50),
            device=config['device']
        )
        
        return env

def _render_frame(obs: Dict, reward: float, task_desc: str = None) -> Optional[np.ndarray]:
    """動画用フレームをレンダリング"""
    try:
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
            cv2.putText(combined, f"Reward: {reward:.2f}", 
                       (10, combined.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # タスク記述を表示
            if task_desc:
                cv2.putText(combined, task_desc, 
                           (10, combined.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            return combined
    except Exception as e:
        logging.warning(f"Failed to render frame: {e}")
    
    return None

def _upload_video(frames: list, epoch: int) -> None:
    """動画をWandBにアップロード"""
    if not frames:
        return
    
    try:
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
    except Exception as e:
        logging.warning(f"Failed to upload video: {e}")

def record_training_video(env, policy, config: Dict, epoch: int) -> None:
    """学習中の動画を記録してWandBにアップロード"""
    try:
        # 環境をリセット
        obs, info = env.reset()
        task_desc = env.get_task_description()
        
        frames = []
        done = False
        episode_reward = 0.0
        episode_length = 0
        max_steps = getattr(env, 'max_episode_steps', 100)
        
        device = config.get('device', 'cuda')
        
        while not done and episode_length < max_steps:
            # フレームを記録
            if hasattr(env, 'current_obs'):
                current_obs = env.current_obs
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
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = policy.actor(obs_tensor)[0].cpu().numpy()
            
            # 環境をステップ実行
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        # 最後のフレームも記録
        if hasattr(env, 'current_obs'):
            current_obs = env.current_obs
            frame = _render_frame(current_obs, episode_reward, task_desc)
            if frame is not None:
                frames.append(frame)
        
        # 動画をアップロード
        if frames:
            _upload_video(frames, epoch)
            logging.info(f"Recorded training video with {len(frames)} frames for epoch {epoch}")
        else:
            logging.warning(f"No frames recorded for epoch {epoch}")
            
    except Exception as e:
        logging.warning(f"Failed to record training video for epoch {epoch}: {e}")

def save_checkpoint(policy, epoch: int, checkpoint_dir: str):
    """チェックポイントを保存"""
    checkpoint_path = Path(checkpoint_dir) / f"epoch_{epoch}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Save policy
    torch.save(policy.state_dict(), checkpoint_path / "policy.pth")
    
    logging.info(f"Checkpoint saved to {checkpoint_path}")

def main(config: Dict):
    """メイン学習関数（動画記録機能付き）"""
    
    # WandBの初期化を先に行う
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
    if config['task'] == 'pendulum':
        # Pendulum環境の場合は並列化可能
        train_envs = DummyVectorEnv([
            lambda: make_env(config) for _ in range(config.get('train_num', 1))
        ])
        test_envs = DummyVectorEnv([
            lambda: make_env(config) for _ in range(config.get('test_num', 1))
        ])
        # Set action space for policy
        env_single = make_env(config)
        action_space = env_single.action_space
        state_dim = env_single.observation_space.shape[0]
        action_dim = action_space.shape[0]
        env_single.close()
        shared_env = None
    else:
        # Genesis環境の場合は単一環境インスタンスを共有
        logging.info("Using single shared environment instance for Genesis-based tasks")
        shared_env = make_env(config)  # 1つの環境インスタンスのみ作成
        
        # 同じ環境インスタンスを使用
        train_envs = DummyVectorEnv([lambda: shared_env])
        test_envs = DummyVectorEnv([lambda: shared_env])
        
        # Set action space for policy (既に作成済みの環境を使用)
        action_space = shared_env.action_space
        state_dim = shared_env.observation_space.shape[0]
        action_dim = action_space.shape[0]
    
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
    test_collector = Collector(policy, test_envs, exploration_noise=False)
    
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
        if (config.get('record_video', False) and config['task'] != 'pendulum' and shared_env is not None):
            record_training_video(shared_env, policy, config, epoch)
    
    # Training - Genesis環境での制限を考慮
    result = offpolicy_trainer(
        policy,
        train_collector,
        None if config['task'] != 'pendulum' else test_collector,  # Genesis環境ではテストコレクターを無効化
        max_epoch=config.get('max_epoch', 100),
        step_per_epoch=config.get('step_per_epoch', 10000),
        step_per_collect=config.get('step_per_collect', 10),
        episode_per_test=config.get('episode_per_test', 10) if config['task'] == 'pendulum' else 0,
        batch_size=config.get('batch_size', 256),
        update_per_step=config.get('update_per_step', 0.1),
        test_in_train=False,  # Genesis環境では常にFalse
        logger=logger,
        save_checkpoint_fn=save_fn,
        resume_from_log=config.get('resume_from_log', False),
        verbose=True
    )
    
    # Clean up
    if config['task'] == 'pendulum':
        train_envs.close()
        test_envs.close()
    else:
        # Genesis環境の場合は共有環境をクリーンアップ
        train_envs.close()
        # test_envsは同じ環境を参照しているので、追加のcloseは不要
        if shared_env is not None:
            shared_env.close()
    
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
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train DSRL with Tianshou SAC")
    parser.add_argument('--eval', action='store_true', help='Run evaluation mode')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint for evaluation')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes for evaluation')
    
    # Configuration overrides
    parser.add_argument('--task', type=str, default='pendulum', help='Task name (pendulum for testing, simple_pick for DSRL)')
    parser.add_argument('--pretrained_model_path', type=str, 
                       default="outputs/train/smolvla_test_0/checkpoints/last/pretrained_model",
                       help='Path to pretrained SmolVLA model')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--show_viewer', action='store_true', help='Show environment viewer')

    # Video recording options
    parser.add_argument('--no-video', action='store_true', help='Disable video recording during training')

    args = parser.parse_args()

    # Configuration
    config = {
        # Environment settings
        'task': args.task,
        'observation_height': 512,
        'observation_width': 512,
        'show_viewer': args.show_viewer,

        # Training settings (adjusted for Genesis limitations)
        'max_epoch': 1000,
        'step_per_epoch': 300,
        'step_per_collect': 10,
        'episode_per_test': 1,
        'batch_size': 16,
        'update_per_step': 0.1,
        'test_in_train': False,

        # Environment parallelization
        'train_num': 1,  # Number of parallel training environments
        'test_num': 1,   # Number of parallel test environments

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
        'use_wandb': not args.no_wandb,
        'wandb_project': 'smolvla',
        'wandb_run_name': f"dsrl_sac_{args.task}",
        'checkpoint_dir': 'outputs/train/dsrl_tianshou_checkpoints',
        'resume_from_log': False,
        'log_per_epoch': 1,

        # Video recording settings
        'record_video': not args.no_video,  # Enable video recording during training

        # SmolVLA settings
        'pretrained_model_path': args.pretrained_model_path,
        'smolvla_config_overrides': {
            'n_action_steps': 10,
        }
    }

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.eval:
        if not args.checkpoint_path:
            parser.error("--checkpoint_path is required for evaluation")
        evaluate_policy(args.checkpoint_path, config, args.num_episodes)
    else:
        main(config)
