# uv run -m src.rl_smolvla
"""
PPO-style reinforcement learning for SmolVLA policy.
This implementation maintains SmolVLA's native action generation while applying PPO-style updates.
Similar to RLHF approaches for LLMs, we collect trajectories using the policy's native sampling
and then update using PPO objectives.
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


class SmolVLATrajectoryBuffer:
    """
    Buffer for storing complete trajectories with SmolVLA's multi-step actions.
    Unlike traditional PPO buffers, this stores complete episodes with SmolVLA's native action chunks.
    """
    def __init__(self, max_episodes: int, gamma: float = 0.99, lam: float = 0.95):
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.lam = lam
        
        # Store complete episodes
        self.episodes = []
        self.current_episode = None
        
    def start_episode(self, initial_obs: Dict, task: str):
        """Start a new episode."""
        self.current_episode = {
            'observations': [initial_obs],
            'actions': [],
            'rewards': [],
            'action_logprobs': [],
            'values': [],
            'dones': [],
            'task': task
        }
    
    def add_step(self, obs: Dict, action: np.ndarray, reward: float, 
                 action_logprob: float, value: float, done: bool):
        """Add a step to the current episode."""
        if self.current_episode is None:
            raise ValueError("Must call start_episode first")
            
        self.current_episode['observations'].append(obs)
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['action_logprobs'].append(action_logprob)
        self.current_episode['values'].append(value)
        self.current_episode['dones'].append(done)
    
    def finish_episode(self, final_value: float = 0.0):
        """Finish the current episode and compute advantages."""
        if self.current_episode is None:
            return
            
        # Compute returns and advantages using GAE
        rewards = np.array(self.current_episode['rewards'])
        values = np.array(self.current_episode['values'])
        
        # Add final value for GAE computation
        values_with_final = np.append(values, final_value)
        
        # Compute GAE advantages
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_with_final[t + 1] - values_with_final[t]
            gae = delta + self.gamma * self.lam * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        self.current_episode['advantages'] = advantages
        self.current_episode['returns'] = returns
        
        # Store episode
        self.episodes.append(self.current_episode)
        
        # Keep only recent episodes
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
        
        self.current_episode = None
    
    def get_all_data(self) -> Dict:
        """Get all stored episode data for training."""
        if not self.episodes:
            return {}
        
        # Flatten all episodes
        all_obs = []
        all_actions = []
        all_returns = []
        all_advantages = []
        all_logprobs = []
        all_tasks = []
        
        for episode in self.episodes:
            # Skip the last observation (no corresponding action)
            episode_obs = episode['observations'][:-1]
            all_obs.extend(episode_obs)
            all_actions.extend(episode['actions'])
            all_returns.extend(episode['returns'])
            all_advantages.extend(episode['advantages'])
            all_logprobs.extend(episode['action_logprobs'])
            all_tasks.extend([episode['task']] * len(episode['actions']))
        
        # Normalize advantages
        advantages = np.array(all_advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'observations': all_obs,
            'actions': np.array(all_actions),
            'returns': np.array(all_returns),
            'advantages': advantages,
            'old_logprobs': np.array(all_logprobs),
            'tasks': all_tasks
        }
    
    def clear(self):
        """Clear all stored episodes."""
        self.episodes.clear()
        self.current_episode = None


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
    SmolVLA policy wrapper that maintains native action generation.
    This class preserves SmolVLA's original flow matching sampling while adding
    the ability to compute log probabilities for PPO updates.
    """
    def __init__(self, smolvla_policy: SmolVLAPolicy):
        self.smolvla_policy = smolvla_policy
        
    def to(self, device):
        """Move policy to device."""
        self.smolvla_policy.to(device)
        return self
    
    def parameters(self):
        """Get all parameters for optimization."""
        return self.smolvla_policy.parameters()
    
    def state_dict(self):
        """Get state dict."""
        return self.smolvla_policy.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.smolvla_policy.load_state_dict(state_dict)
    
    def train(self):
        """Set to training mode."""
        self.smolvla_policy.train()
    
    def eval(self):
        """Set to evaluation mode."""
        self.smolvla_policy.eval()
    
    def sample_action(self, batch: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, float]:
        """
        Sample action using SmolVLA's native sampling with noise.
        Returns action and a placeholder log probability (will be computed separately).
        """
        self.eval()
        
        with torch.no_grad():
            if deterministic:
                # Use deterministic action (no noise)
                action = self.smolvla_policy.select_action(batch, noise=None)
            else:
                # Use SmolVLA's native sampling with noise
                action = self.smolvla_policy.select_action(batch)
        
        # Placeholder log probability - will be computed in evaluate_action
        log_prob = 0.0
        
        return action, log_prob
    
    def evaluate_action(self, batch: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[float, float]:
        """
        Evaluate an action by computing the negative loss as a proxy for log probability.
        This is similar to how language models compute log probabilities for RLHF.
        """
        self.train()
        
        # Create a batch with the action for forward pass
        eval_batch = batch.copy()
        eval_batch['action'] = action.unsqueeze(0) if action.dim() == 1 else action
        
        # Compute the loss (negative log likelihood)
        loss, _ = self.smolvla_policy.forward(eval_batch)
        
        # Use negative loss as proxy for log probability
        # This is conceptually similar to how we handle discrete tokens in language models
        log_prob = -loss.item()
        
        # Compute entropy as a regularization term
        # For continuous actions, we can estimate this from the policy's internal noise
        entropy = 0.1  # Placeholder - could be made more sophisticated
        
        return log_prob, entropy


class SmolVLARLTrainer:
    """
    RL trainer for SmolVLA that maintains native action generation.
    Uses trajectory-level updates similar to RLHF for language models.
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
        
        # Initialize trajectory buffer
        self.buffer = SmolVLATrajectoryBuffer(
            max_episodes=config['max_episodes_in_buffer'],
            gamma=config['gamma'],
            lam=config['lam']
        )
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize wandb
        wandb.init(
            project=config.get('wandb_project', 'smolvla-rl'),
            name=config.get('wandb_run_name', f"rl_smolvla_{config['task']}"),
            config=config,
            sync_tensorboard=False
        )
        
        # Training metrics
        self.epoch = 0
        self.total_episodes = 0
        
        # Video recording
        self.video_frames = []
        self.record_video = config.get('record_video', True)
        self.video_freq = config.get('video_freq', 10)
        
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
    
    def collect_episodes(self) -> Dict:
        """
        Collect complete episodes using SmolVLA's native action generation.
        This maintains the policy's natural behavior while collecting data for RL updates.
        """
        episode_rewards = []
        episode_lengths = []
        
        episodes_collected = 0
        target_episodes = self.config['episodes_per_update']
        
        while episodes_collected < target_episodes:
            # Start new episode
            obs, _ = self.env.reset()
            task = task_description.get(self.env.task, "Complete the task")
            
            self.buffer.start_episode(obs, task)
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Start video recording if enabled
            if self.record_video and (self.epoch % self.video_freq == 0) and episodes_collected == 0:
                self.video_frames = []
                frame = self._render_combined_frame(obs)
                if frame is not None:
                    self.video_frames.append(frame)
            
            # Reset policy's internal state
            self.policy.smolvla_policy.reset()
            
            while not done:
                # Prepare batch for policy
                batch = self._prepare_batch_for_policy(obs, task)
                
                # Sample action using SmolVLA's native sampling
                with torch.no_grad():
                    action, _ = self.policy.sample_action(batch, deterministic=False)
                    # Get value estimate
                    value = self.value_function(batch['observation.state'])
                
                # Convert to numpy for environment
                action_np = action.cpu().numpy().squeeze()
                value_np = value.cpu().numpy().item()
                
                # Execute action sequence (SmolVLA generates multiple steps)
                step_rewards = []
                for step_idx in range(self.policy.smolvla_policy.config.n_action_steps):
                    if step_idx < len(action_np):
                        single_action = action_np[step_idx] if action_np.ndim > 1 else action_np
                    else:
                        single_action = action_np[-1] if action_np.ndim > 1 else action_np
                    
                    next_obs, reward, terminated, truncated, info = self.env.step(single_action)
                    step_rewards.append(reward)
                    episode_reward += reward
                    episode_length += 1
                    
                    # Record video frame
                    if (self.record_video and (self.epoch % self.video_freq == 0) and 
                        episodes_collected == 0):
                        frame = self._render_combined_frame(next_obs)
                        if frame is not None:
                            self.video_frames.append(frame)
                    
                    if terminated or truncated:
                        done = True
                        break
                    
                    obs = next_obs
                
                # Compute log probability for the entire action sequence
                # This is done after execution to avoid interfering with native sampling
                with torch.no_grad():
                    log_prob, _ = self.policy.evaluate_action(batch, action)
                
                # Store the complete action sequence and cumulative reward
                total_step_reward = sum(step_rewards)
                self.buffer.add_step(
                    obs=next_obs if not done else obs,
                    action=action_np,
                    reward=total_step_reward,
                    action_logprob=log_prob,
                    value=value_np,
                    done=done
                )
            
            # Finish episode
            if done:
                final_value = 0.0
            else:
                with torch.no_grad():
                    final_batch = self._prepare_batch_for_policy(obs, task)
                    final_value = self.value_function(final_batch['observation.state']).cpu().numpy().item()
            
            self.buffer.finish_episode(final_value)
            
            # Record episode stats
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episodes_collected += 1
            self.total_episodes += 1
            
            self.logger.info(
                f"Episode {self.total_episodes}: Reward={episode_reward:.2f}, Length={episode_length}"
            )
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths)
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
        """Update policy using PPO-style objective on trajectory data."""
        if not data:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
        
        observations = data['observations']
        actions = torch.from_numpy(data['actions']).float().to(self.device)
        returns = torch.from_numpy(data['returns']).float().to(self.device)
        advantages = torch.from_numpy(data['advantages']).float().to(self.device)
        old_logprobs = torch.from_numpy(data['old_logprobs']).float().to(self.device)
        
        # Prepare batches for policy evaluation
        batches = []
        for i, obs in enumerate(observations):
            batch_i = self._prepare_batch_for_policy(obs, data['tasks'][i])
            batches.append(batch_i)
        
        # Policy updates
        policy_losses = []
        value_losses = []
        entropies = []
        
        for update_epoch in range(self.config['update_epochs']):
            # Shuffle data for each epoch
            indices = torch.randperm(len(observations))
            
            for batch_start in range(0, len(observations), self.config['batch_size']):
                batch_end = min(batch_start + self.config['batch_size'], len(observations))
                batch_indices = indices[batch_start:batch_end]
                
                if len(batch_indices) == 0:
                    continue
                
                # Get batch data
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_observations = [batches[i] for i in batch_indices]
                
                # Update policy
                self.policy_optimizer.zero_grad()
                
                # Evaluate actions with current policy
                current_logprobs = []
                current_entropies = []
                
                for i, obs_batch in enumerate(batch_observations):
                    action_idx = batch_indices[i]
                    action = actions[action_idx]
                    
                    log_prob, entropy = self.policy.evaluate_action(obs_batch, action)
                    current_logprobs.append(log_prob)
                    current_entropies.append(entropy)
                
                current_logprobs = torch.tensor(current_logprobs, device=self.device)
                current_entropies = torch.tensor(current_entropies, device=self.device)
                
                # Compute policy loss (PPO clipped objective)
                ratio = torch.exp(current_logprobs - batch_old_logprobs)
                clipped_ratio = torch.clamp(ratio, 1 - self.config['clip_ratio'], 1 + self.config['clip_ratio'])
                
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                entropy_loss = -self.config['entropy_coeff'] * current_entropies.mean()
                
                total_policy_loss = policy_loss + entropy_loss
                total_policy_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['max_grad_norm'])
                
                self.policy_optimizer.step()
                policy_losses.append(total_policy_loss.item())
                entropies.append(current_entropies.mean().item())
                
                # Update value function
                self.value_optimizer.zero_grad()
                
                # Compute state values for batch observations
                state_values = []
                for obs_batch in batch_observations:
                    value = self.value_function(obs_batch['observation.state'])
                    state_values.append(value)
                
                state_values = torch.stack(state_values).squeeze()
                value_loss = F.mse_loss(state_values, batch_returns)
                
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), self.config['max_grad_norm'])
                
                self.value_optimizer.step()
                value_losses.append(value_loss.item())
        
        return {
            'policy_loss': np.mean(policy_losses) if policy_losses else 0,
            'value_loss': np.mean(value_losses) if value_losses else 0,
            'entropy': np.mean(entropies) if entropies else 0
        }
    
    def train(self):
        """Main training loop with wandb logging and video upload."""
        self.logger.info(f"Starting RL training for {self.config['epochs']} epochs")
        
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            
            # Collect episodes
            self.logger.info(f"Epoch {epoch}: Collecting episodes...")
            episode_stats = self.collect_episodes()
            
            # Get trajectory data
            data = self.buffer.get_all_data()
            
            # Update policy
            if data:
                self.logger.info(f"Epoch {epoch}: Updating policy...")
                update_stats = self.update_policy(data)
            else:
                update_stats = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
            
            # Create log dictionary
            log_dict = {
                'epoch': epoch,
                'reward/mean': episode_stats['mean_reward'],
                'reward/std': episode_stats['std_reward'],
                'episode/length': episode_stats['mean_length'],
                'loss/policy': update_stats['policy_loss'],
                'loss/value': update_stats['value_loss'],
                'policy/entropy': update_stats['entropy'],
                'training/total_episodes': self.total_episodes
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
                f"Mean Reward: {episode_stats['mean_reward']:.2f} ± {episode_stats['std_reward']:.2f}, "
                f"Mean Length: {episode_stats['mean_length']:.1f}, "
                f"Policy Loss: {update_stats['policy_loss']:.4f}, "
                f"Value Loss: {update_stats['value_loss']:.4f}"
            )
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch)
            
            # Clear buffer periodically to manage memory
            if (epoch + 1) % self.config.get('buffer_clear_freq', 10) == 0:
                self.buffer.clear()
        
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
        
        # RL hyperparameters
        'epochs': 1000,
        'episodes_per_update': 10,  # Collect 10 episodes before each update
        'max_episodes_in_buffer': 50,  # Keep last 50 episodes in buffer
        'gamma': 0.99,
        'lam': 0.95,
        'clip_ratio': 0.2,
        'policy_lr': 1e-4,  # Lower LR for SmolVLA fine-tuning
        'value_lr': 3e-4,
        'update_epochs': 4,  # Number of update epochs per training iteration
        'batch_size': 8,  # Batch size for updates
        'entropy_coeff': 0.01,
        'max_grad_norm': 0.5,
        
        # Logging and saving
        'wandb_project': 'smolvla-rl',
        'wandb_run_name': None,  # Will be auto-generated
        'checkpoint_dir': 'outputs/rl_checkpoints',
        'save_freq': 20,
        'buffer_clear_freq': 50,  # Clear buffer every N epochs to manage memory
        
        # Video recording
        'record_video': True,
        'video_freq': 10,  # Record video every 10 epochs
        'video_fps': 30,
        
        # Model
        'pretrained_model_path': "outputs/train/smolvla_test_0/checkpoints/last/pretrained_model",
        'hidden_dim': 256,
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
    
    # Create direct policy wrapper
    rl_policy = SmolVLADirectPolicy(smolvla_policy)
    
    # Create value function
    obs_dim = env.observation_space['agent_pos'].shape[0]
    value_function = PPOValueFunction(obs_dim, config['hidden_dim'])
    
    # Create trainer
    trainer = SmolVLARLTrainer(
        env=env,
        policy=rl_policy,
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
