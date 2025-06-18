# uv run -m src.rl_smolvla
"""
PPO (Proximal Policy Optimization) training for SmolVLA policy.
This script implements on-policy reinforcement learning using PPO to fine-tune SmolVLA policy.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import math
import wandb
import imageio
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.genesis_env import GenesisEnv
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from src.make_sim_dataset import task_description


class PPOBuffer:
    """
    Buffer for storing rollout data for PPO training.
    """
    def __init__(self, obs_dim: int, act_dim: int, size: int, gamma: float = 0.99, lam: float = 0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        
        # Store image observations separately
        self.front_img_buf = []
        self.side_img_buf = []
        self.task_buf = []
        
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size

    def store(self, obs: Dict, act: np.ndarray, rew: float, val: float, logp: float):
        """Store a single transition."""
        assert self.ptr < self.max_size
        
        # Store state observation
        self.obs_buf[self.ptr] = obs['agent_pos']
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        
        # Store image observations
        self.front_img_buf.append(obs['observation.images.front'])
        self.side_img_buf.append(obs['observation.images.side'])
        self.task_buf.append(obs.get('task', ''))
        
        self.ptr += 1

    def finish_path(self, last_val: float = 0):
        """Finish trajectory and compute GAE-Lambda advantage estimates."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        
        # Rewards-to-go (targets for value function)
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self) -> Dict:
        """Get all stored data and reset buffer."""
        assert self.ptr == self.max_size
        self.ptr = 0
        self.path_start_idx = 0
        
        # Normalize advantages
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        data = {
            'obs': self.obs_buf,
            'act': self.act_buf,
            'ret': self.ret_buf,
            'adv': self.adv_buf,
            'logp': self.logp_buf,
            'front_img': self.front_img_buf.copy(),
            'side_img': self.side_img_buf.copy(),
            'task': self.task_buf.copy()
        }
        
        # Clear image buffers
        self.front_img_buf.clear()
        self.side_img_buf.clear()
        self.task_buf.clear()
        
        return data

    def _discount_cumsum(self, x: np.ndarray, discount: float) -> np.ndarray:
        """Compute discounted cumulative sum."""
        return np.array([np.sum(discount**np.arange(len(x)-i) * x[i:]) for i in range(len(x))])


class PPOValueFunction(nn.Module):
    """
    Value function network for PPO.
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class SmolVLADirectPolicy:
    """
    Direct SmolVLA policy wrapper for PPO training.
    This maintains the original SmolVLA architecture without additional action heads.
    """
    def __init__(self, smolvla_policy: SmolVLAPolicy, action_std: float = 0.1):
        self.smolvla_policy = smolvla_policy
        self.action_std = action_std
        # Create learnable log_std parameter
        self.log_std = nn.Parameter(torch.full((smolvla_policy.config.max_action_dim,), math.log(action_std)))
        
    def to(self, device):
        """Move policy to device."""
        self.smolvla_policy.to(device)
        self.log_std = self.log_std.to(device)
        return self
    
    def parameters(self):
        """Get all parameters for optimization."""
        for param in self.smolvla_policy.parameters():
            yield param
        yield self.log_std
    
    def state_dict(self):
        """Get state dict."""
        return {
            'smolvla_policy': self.smolvla_policy.state_dict(),
            'log_std': self.log_std
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.smolvla_policy.load_state_dict(state_dict['smolvla_policy'])
        self.log_std = state_dict['log_std']
    
    def get_action_and_logprob(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability using SmolVLA directly."""
        # Get action from SmolVLA (deterministic)
        action_mean = self.smolvla_policy.select_action(batch)
        
        # Create Gaussian distribution
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        
        # Sample action
        action = dist.sample()
        logprob = dist.log_prob(action).sum(axis=-1)
        
        return action, logprob
    
    def evaluate_actions(self, batch: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate actions and return log probabilities and entropy."""
        # Get action mean from SmolVLA
        action_mean = self.smolvla_policy.select_action(batch)
        
        # Create Gaussian distribution
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        
        # Compute log probability and entropy
        logprob = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        
        return logprob, entropy


class PPOTrainer:
    """
    PPO trainer for SmolVLA policy with wandb logging and video recording.
    """
    def __init__(
        self,
        env: GenesisEnv,
        policy: SmolVLADirectPolicy,
        value_function: PPOValueFunction,
        config: Dict,
        device: str = "cuda"
    ):
        self.env = env
        self.policy = policy
        self.value_function = value_function
        self.config = config
        self.device = device
        
        # Move models to device
        self.policy.to(device)
        self.value_function.to(device)
        
        # Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=config['policy_lr']
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_function.parameters(), 
            lr=config['value_lr']
        )
        
        # Initialize buffer
        obs_dim = env.observation_space['agent_pos'].shape[0]
        act_dim = env.action_space.shape[0]
        self.buffer = PPOBuffer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            size=config['steps_per_epoch'],
            gamma=config['gamma'],
            lam=config['lam']
        )
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize wandb
        wandb.init(
            project=config.get('wandb_project', 'smolvla-ppo'),
            name=config.get('wandb_run_name', f"ppo_smolvla_{config['task']}"),
            config=config,
            sync_tensorboard=False
        )
        
        # Training metrics
        self.epoch = 0
        self.total_steps = 0
        
        # Video recording
        self.video_frames = []
        self.record_video = config.get('record_video', True)
        self.video_freq = config.get('video_freq', 10)  # Record video every N epochs
        
    def _prepare_batch_for_policy(self, obs: Dict, task: str) -> Dict[str, torch.Tensor]:
        """Prepare observation batch for policy inference."""
        batch = {}
        
        # State observation
        if 'agent_pos' in obs:
            batch['observation.state'] = torch.from_numpy(obs['agent_pos']).float().to(self.device).unsqueeze(0)
        
        # Image observations
        if 'observation.images.front' in obs:
            front_img = obs['observation.images.front'].copy()
            tensor_img = torch.from_numpy(front_img).float().to(self.device) / 255.0
            if tensor_img.ndim == 3 and tensor_img.shape[2] in [1, 3, 4]:
                tensor_img = tensor_img.permute(2, 0, 1)
            elif tensor_img.ndim == 2:
                tensor_img = tensor_img.unsqueeze(0)
            batch['observation.images.front'] = tensor_img.unsqueeze(0)
        
        if 'observation.images.side' in obs:
            side_img = obs['observation.images.side'].copy()
            tensor_img = torch.from_numpy(side_img).float().to(self.device) / 255.0
            if tensor_img.ndim == 3 and tensor_img.shape[2] in [1, 3, 4]:
                tensor_img = tensor_img.permute(2, 0, 1)
            elif tensor_img.ndim == 2:
                tensor_img = tensor_img.unsqueeze(0)
            batch['observation.images.side'] = tensor_img.unsqueeze(0)
        
        # Task description
        batch['task'] = task
        
        return batch
    
    def collect_rollout(self) -> Dict:
        """Collect rollout data for PPO training with video recording."""
        obs, _ = self.env.reset()
        task = task_description.get(self.env.task, "Complete the task")
        
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        # Start video recording if enabled
        if self.record_video and (self.epoch % self.video_freq == 0):
            self.video_frames = []
            # Record initial frame
            frame = self._render_combined_frame(obs)
            if frame is not None:
                self.video_frames.append(frame)
        
        for step in range(self.config['steps_per_epoch']):
            # Prepare batch for policy
            batch = self._prepare_batch_for_policy(obs, task)
            
            # Get action and value
            with torch.no_grad():
                action, logprob = self.policy.get_action_and_logprob(batch)
                value = self.value_function(batch['observation.state'])
            
            # Convert to numpy
            action_np = action.cpu().numpy().squeeze()
            logprob_np = logprob.cpu().numpy().item()
            value_np = value.cpu().numpy().item()
            
            # Store in buffer
            obs_with_task = obs.copy()
            obs_with_task['task'] = task
            self.buffer.store(obs_with_task, action_np, 0, value_np, logprob_np)  # reward will be updated
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            
            # Update buffer with actual reward
            self.buffer.rew_buf[self.buffer.ptr - 1] = reward
            
            current_episode_reward += reward
            current_episode_length += 1
            
            # Record video frame
            if self.record_video and (self.epoch % self.video_freq == 0):
                frame = self._render_combined_frame(next_obs)
                if frame is not None:
                    self.video_frames.append(frame)
            
            # Handle episode termination
            if terminated or truncated:
                # Finish path in buffer
                self.buffer.finish_path(0)
                
                # Log episode metrics
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                
                # Reset environment
                obs, _ = self.env.reset()
                task = task_description.get(self.env.task, "Complete the task")
                current_episode_reward = 0
                current_episode_length = 0
                
                # Record reset frame for video
                if self.record_video and (self.epoch % self.video_freq == 0):
                    frame = self._render_combined_frame(obs)
                    if frame is not None:
                        self.video_frames.append(frame)
            else:
                obs = next_obs
            
            self.total_steps += 1
        
        # Finish any remaining path
        if self.buffer.path_start_idx < self.buffer.ptr:
            with torch.no_grad():
                batch = self._prepare_batch_for_policy(obs, task)
                last_value = self.value_function(batch['observation.state']).cpu().numpy().item()
            self.buffer.finish_path(last_value)
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'std_reward': np.std(episode_rewards) if episode_rewards else 0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0
        }
    
    def _render_combined_frame(self, obs: Dict) -> Optional[np.ndarray]:
        """Create combined frame from front and side camera observations."""
        try:
            front_img = obs.get('observation.images.front')
            side_img = obs.get('observation.images.side')
            
            if front_img is None or side_img is None:
                return None
            
            # Ensure images are in the right format (H, W, C) and uint8
            if front_img.dtype != np.uint8:
                front_img = (front_img * 255).astype(np.uint8) if front_img.dtype == np.float32 else front_img
            if side_img.dtype != np.uint8:
                side_img = (side_img * 255).astype(np.uint8) if side_img.dtype == np.float32 else side_img
            
            # Concatenate horizontally
            combined_frame = np.concatenate([front_img, side_img], axis=1)
            return combined_frame
            
        except Exception as e:
            self.logger.warning(f"Failed to render combined frame: {e}")
            return None
    
    def update_policy(self, data: Dict) -> Dict:
        """Update policy using PPO."""
        obs = torch.from_numpy(data['obs']).float().to(self.device)
        actions = torch.from_numpy(data['act']).float().to(self.device)
        returns = torch.from_numpy(data['ret']).float().to(self.device)
        advantages = torch.from_numpy(data['adv']).float().to(self.device)
        old_logprobs = torch.from_numpy(data['logp']).float().to(self.device)
        
        # Prepare batch for policy evaluation
        batch_list = []
        for i in range(len(data['obs'])):
            batch_i = {
                'observation.state': obs[i:i+1],
                'task': data['task'][i] if data['task'][i] else "Complete the task"
            }
            
            # Add images if available
            if data['front_img'][i] is not None:
                front_img = torch.from_numpy(data['front_img'][i]).float().to(self.device) / 255.0
                if front_img.ndim == 3 and front_img.shape[2] in [1, 3, 4]:
                    front_img = front_img.permute(2, 0, 1)
                elif front_img.ndim == 2:
                    front_img = front_img.unsqueeze(0)
                batch_i['observation.images.front'] = front_img.unsqueeze(0)
            
            if data['side_img'][i] is not None:
                side_img = torch.from_numpy(data['side_img'][i]).float().to(self.device) / 255.0
                if side_img.ndim == 3 and side_img.shape[2] in [1, 3, 4]:
                    side_img = side_img.permute(2, 0, 1)
                elif side_img.ndim == 2:
                    side_img = side_img.unsqueeze(0)
                batch_i['observation.images.side'] = side_img.unsqueeze(0)
            
            batch_list.append(batch_i)
        
        # Policy update
        policy_losses = []
        value_losses = []
        
        for _ in range(self.config['train_pi_iters']):
            self.policy_optimizer.zero_grad()
            
            # Evaluate actions with current policy
            logprobs_list = []
            entropies_list = []
            
            for i, batch_i in enumerate(batch_list):
                logprob, entropy = self.policy.evaluate_actions(batch_i, actions[i:i+1])
                logprobs_list.append(logprob)
                entropies_list.append(entropy)
            
            logprobs = torch.cat(logprobs_list)
            entropies = torch.cat(entropies_list)
            
            # Compute policy loss
            ratio = torch.exp(logprobs - old_logprobs)
            clipped_ratio = torch.clamp(ratio, 1 - self.config['clip_ratio'], 1 + self.config['clip_ratio'])
            
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            entropy_loss = -self.config['entropy_coeff'] * entropies.mean()
            
            total_policy_loss = policy_loss + entropy_loss
            total_policy_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['max_grad_norm'])
            
            self.policy_optimizer.step()
            policy_losses.append(total_policy_loss.item())
        
        # Value function update
        for _ in range(self.config['train_v_iters']):
            self.value_optimizer.zero_grad()
            
            values = self.value_function(obs)
            value_loss = F.mse_loss(values, returns)
            
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), self.config['max_grad_norm'])
            
            self.value_optimizer.step()
            value_losses.append(value_loss.item())
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': entropies.mean().item()
        }
    
    def train(self):
        """Main training loop with wandb logging and video upload."""
        self.logger.info(f"Starting PPO training for {self.config['epochs']} epochs")
        
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            
            # Collect rollout data
            self.logger.info(f"Epoch {epoch}: Collecting rollouts...")
            rollout_stats = self.collect_rollout()
            
            # Get buffer data
            data = self.buffer.get()
            
            # Update policy
            self.logger.info(f"Epoch {epoch}: Updating policy...")
            update_stats = self.update_policy(data)
            
            # Create log dictionary
            log_dict = {
                'epoch': epoch,
                'reward/mean': rollout_stats['mean_reward'],
                'reward/std': rollout_stats['std_reward'],
                'episode/length': rollout_stats['mean_length'],
                'loss/policy': update_stats['policy_loss'],
                'loss/value': update_stats['value_loss'],
                'policy/entropy': update_stats['entropy'],
                'policy/action_std': torch.exp(self.policy.log_std).mean().item(),
                'training/total_steps': self.total_steps
            }
            
            # Upload video to wandb
            if self.record_video and (epoch % self.video_freq == 0) and self.video_frames:
                try:
                    # Create video from frames
                    video_path = Path(self.config['checkpoint_dir']) / f"video_epoch_{epoch}.mp4"
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save video
                    imageio.mimsave(
                        str(video_path), 
                        self.video_frames, 
                        fps=self.config.get('video_fps', 30),
                        output_params=['-pix_fmt', 'yuv420p']
                    )
                    
                    # Upload to wandb
                    log_dict['video/rollout'] = wandb.Video(str(video_path), fps=self.config.get('video_fps', 30))
                    
                    self.logger.info(f"Video saved and uploaded: {video_path}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to save/upload video: {e}")
            
            # Log to wandb
            wandb.log(log_dict, step=epoch)
            
            # Console logging
            self.logger.info(
                f"Epoch {epoch}: "
                f"Mean Reward: {rollout_stats['mean_reward']:.2f} ± {rollout_stats['std_reward']:.2f}, "
                f"Mean Length: {rollout_stats['mean_length']:.1f}, "
                f"Policy Loss: {update_stats['policy_loss']:.4f}, "
                f"Value Loss: {update_stats['value_loss']:.4f}, "
                f"Action Std: {torch.exp(self.policy.log_std).mean().item():.4f}"
            )
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch)
        
        self.logger.info("Training completed!")
        wandb.finish()
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoints."""
        checkpoint_dir = Path(self.config['checkpoint_dir']) / f"epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save policy state dict
        policy_path = checkpoint_dir / "policy.pth"
        torch.save({
            'epoch': epoch,
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_function.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': self.config
        }, policy_path)
        
        # Save SmolVLA policy separately
        smolvla_path = checkpoint_dir / "smolvla_policy"
        self.policy.smolvla_policy.save_pretrained(smolvla_path)
        
        # Upload checkpoint to wandb
        wandb.save(str(policy_path))
        
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")


def main():
    """Main training function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Training configuration
    config = {
        # Environment
        'task': 'test',
        'observation_height': 480,
        'observation_width': 640,
        'show_viewer': False,
        
        # PPO hyperparameters
        'epochs': 1000,
        'steps_per_epoch': 4000,
        'gamma': 0.99,
        'lam': 0.95,
        'clip_ratio': 0.2,
        'policy_lr': 3e-4,
        'value_lr': 1e-3,
        'train_pi_iters': 80,
        'train_v_iters': 80,
        'entropy_coeff': 0.01,
        'max_grad_norm': 0.5,
        
        # Logging and saving
        'wandb_project': 'smolvla-ppo',
        'wandb_run_name': None,  # Will be auto-generated
        'checkpoint_dir': 'outputs/ppo_checkpoints',
        'save_freq': 10,
        
        # Video recording
        'record_video': True,
        'video_freq': 5,  # Record video every 5 epochs
        'video_fps': 30,
        
        # Model
        'pretrained_model_path': None,  # Set to load pretrained SmolVLA
        'hidden_dim': 256,
        'action_std': 0.1,  # Initial action standard deviation
    }
    
    # Create output directories
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create environment
    logger.info("Creating environment...")
    env = GenesisEnv(
        task=config['task'],
        observation_height=config['observation_height'],
        observation_width=config['observation_width'],
        show_viewer=config['show_viewer']
    )
    
    # Create SmolVLA policy
    logger.info("Creating SmolVLA policy...")
    if config['pretrained_model_path']:
        smolvla_policy = SmolVLAPolicy.from_pretrained(config['pretrained_model_path'])
    else:
        # Create from scratch with default config
        smolvla_config = SmolVLAConfig()
        smolvla_policy = SmolVLAPolicy(smolvla_config)
    
    # Create direct PPO policy wrapper
    ppo_policy = SmolVLADirectPolicy(smolvla_policy, config['action_std'])
    
    # Create value function
    obs_dim = env.observation_space['agent_pos'].shape[0]
    value_function = PPOValueFunction(obs_dim, config['hidden_dim'])
    
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        policy=ppo_policy,
        value_function=value_function,
        config=config,
        device=device
    )
    
    # Start training
    trainer.train()
    
    # Clean up
    env.close()


if __name__ == "__main__":
    main()
