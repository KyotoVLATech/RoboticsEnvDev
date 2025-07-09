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

def save_checkpoint(policy, epoch: int, checkpoint_dir: str):
    """チェックポイントを保存"""
    checkpoint_path = Path(checkpoint_dir) / f"epoch_{epoch}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Save policy
    torch.save(policy.state_dict(), checkpoint_path / "policy.pth")
    
    logging.info(f"Checkpoint saved to {checkpoint_path}")

def main(config: Dict):
    """メイン学習関数"""
    
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
    
    # Environment作成
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
    
    # Checkpoint function
    def save_fn(epoch, env_step, gradient_step):
        save_checkpoint(policy, epoch, config['checkpoint_dir'])
    
    # Training
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=config.get('max_epoch', 100),
        step_per_epoch=config.get('step_per_epoch', 10000),
        step_per_collect=config.get('step_per_collect', 10),
        episode_per_test=config.get('episode_per_test', 10),
        batch_size=config.get('batch_size', 256),
        update_per_step=config.get('update_per_step', 0.1),
        test_in_train=config.get('test_in_train', False),
        logger=logger,
        save_checkpoint_fn=save_fn,
        resume_from_log=config.get('resume_from_log', False),
        verbose=True
    )
    
    # Clean up
    train_envs.close()
    test_envs.close()
    
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
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        # Environment settings
        'task': args.task,
        'observation_height': 512,
        'observation_width': 512,
        'show_viewer': args.show_viewer,
        
        # Training settings (adjusted for pendulum if needed)
        'max_epoch': 1000,
        'step_per_epoch': 1000,
        'step_per_collect': 10,
        'episode_per_test': 5,
        'batch_size': 64,
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
        
        # SmolVLA settings
        'pretrained_model_path': args.pretrained_model_path,
        'smolvla_config_overrides': {
            'n_action_steps': 20,
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
