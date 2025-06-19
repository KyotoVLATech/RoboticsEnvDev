"""
GRPO (Generalized Relative Preference Optimization) for SmolVLA policy.
This implementation uses preference-based optimization to improve SmolVLA's performance
without requiring value functions, making it more efficient than PPO.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
from dataclasses import dataclass
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.genesis_env import GenesisEnv
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from src.make_sim_dataset import task_description


@dataclass
class Trajectory:
    """Single trajectory with observations, actions, and rewards."""
    observations: List[Dict]
    actions: List[np.ndarray]
    rewards: List[float] 
    task: str
    total_reward: float = 0.0
    
    def __post_init__(self):
        self.total_reward = sum(self.rewards)


class SmolVLAPreferenceBuffer:
    """
    Buffer for storing trajectory pairs with preference labels for GRPO.
    Automatically generates preferences based on cumulative rewards.
    """
    def __init__(self, max_pairs: int = 1000):
        self.max_pairs = max_pairs
        self.trajectories = []
        self.preference_pairs = []
        
    def add_trajectory(self, trajectory: Trajectory):
        """Add a trajectory to the buffer."""
        self.trajectories.append(trajectory)
        
        # Create preference pairs with existing trajectories
        if len(self.trajectories) >= 2:
            self._create_preference_pairs()
            
        # Keep only recent trajectories
        if len(self.trajectories) > self.max_pairs * 2:
            self.trajectories = self.trajectories[-self.max_pairs:]
    
    def _create_preference_pairs(self):
        """Create preference pairs from trajectories based on rewards."""
        if len(self.trajectories) < 2:
            return
            
        # Sample pairs from recent trajectories
        recent_trajectories = self.trajectories[-10:]  # Use last 10 trajectories
        
        for i in range(len(recent_trajectories)):
            for j in range(i + 1, len(recent_trajectories)):
                traj_1 = recent_trajectories[i]
                traj_2 = recent_trajectories[j]
                
                # Create preference based on total reward
                if traj_1.total_reward > traj_2.total_reward:
                    preferred, dispreferred = traj_1, traj_2
                else:
                    preferred, dispreferred = traj_2, traj_1
                
                self.preference_pairs.append((preferred, dispreferred))
        
        # Keep only recent pairs
        if len(self.preference_pairs) > self.max_pairs:
            self.preference_pairs = self.preference_pairs[-self.max_pairs:]
    
    def get_preference_batch(self, batch_size: int) -> List[Tuple[Trajectory, Trajectory]]:
        """Get a batch of preference pairs."""
        if len(self.preference_pairs) < batch_size:
            return self.preference_pairs
        
        return random.sample(self.preference_pairs, batch_size)
    
    def clear(self):
        """Clear all stored data."""
        self.trajectories.clear()
        self.preference_pairs.clear()


class SmolVLAGRPOPolicy:
    """
    SmolVLA policy wrapper for GRPO training.
    Maintains native action generation while enabling preference-based optimization.
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
    
    def sample_action(self, batch: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """Sample action using SmolVLA's native sampling."""
        self.eval()
        
        with torch.no_grad():
            # Use select_action method like in eval_policy.py
            action = self.smolvla_policy.select_action(batch)
            
            # Handle different return formats
            if isinstance(action, dict):
                action_tensor = action.get('action', None)
                if action_tensor is None:
                    raise ValueError("Policy did not return 'action' key.")
            else:
                action_tensor = action
        
        return action_tensor
    
    def compute_trajectory_logprob(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Compute log probability of a trajectory by using the model's loss as proxy.
        This is similar to how RLHF works for language models.
        """
        self.train()
        total_logprob = 0.0
        
        for i, (obs, action) in enumerate(zip(trajectory.observations, trajectory.actions)):
            # Prepare batch
            batch = self._prepare_batch_for_policy(obs, trajectory.task)
            
            try:
                # Create target action tensor
                action_tensor = torch.from_numpy(action).float().to(next(self.parameters()).device)
                if action_tensor.dim() == 1:
                    action_tensor = action_tensor.unsqueeze(0)
                
                # Add actions to batch
                batch['actions'] = action_tensor
                
                # Compute forward pass
                outputs = self.smolvla_policy.forward(batch)
                
                # Use negative loss as log probability proxy
                if hasattr(outputs, 'loss'):
                    step_logprob = -outputs.loss
                else:
                    step_logprob = torch.tensor(0.0, requires_grad=True)
                
                total_logprob += step_logprob
                
            except Exception as e:
                # Fallback to small penalty if computation fails
                total_logprob += torch.tensor(-1.0, requires_grad=True)
        
        return total_logprob
    
    def _prepare_batch_for_policy(self, obs: Dict, task: str) -> Dict[str, torch.Tensor]:
        """Prepare observation batch for policy inference."""
        batch = {}
        device = next(self.parameters()).device
        
        # State observation - check what keys are available in obs
        if 'agent_pos' in obs:
            agent_pos = obs['agent_pos']
            if isinstance(agent_pos, np.ndarray):
                agent_pos = agent_pos.copy()
            agent_pos = torch.from_numpy(agent_pos).float().to(device)
            if agent_pos.dim() == 1:
                agent_pos = agent_pos.unsqueeze(0)
            batch['observation.state'] = agent_pos
        
        # Image observations - follow eval_policy.py format
        if 'observation.images.front' in obs:
            front_img = obs['observation.images.front']
            if isinstance(front_img, np.ndarray):
                # Make a copy to handle negative strides
                front_img = front_img.copy()
                
            # Convert to tensor and normalize to [0, 1]
            front_img = torch.from_numpy(front_img).float() / 255.0
            
            # Handle different image formats
            if front_img.ndim == 3 and front_img.shape[2] in [1, 3, 4]:
                front_img = front_img.permute(2, 0, 1)  # HWC -> CHW
            elif front_img.ndim == 2:
                front_img = front_img.unsqueeze(0)  # Add channel dimension
            
            # Add batch dimension
            batch['observation.images.front'] = front_img.to(device).unsqueeze(0)
        
        if 'observation.images.side' in obs:
            side_img = obs['observation.images.side']
            if isinstance(side_img, np.ndarray):
                # Make a copy to handle negative strides
                side_img = side_img.copy()
                
            # Convert to tensor and normalize to [0, 1]
            side_img = torch.from_numpy(side_img).float() / 255.0
            
            # Handle different image formats
            if side_img.ndim == 3 and side_img.shape[2] in [1, 3, 4]:
                side_img = side_img.permute(2, 0, 1)  # HWC -> CHW
            elif side_img.ndim == 2:
                side_img = side_img.unsqueeze(0)  # Add channel dimension
            
            # Add batch dimension
            batch['observation.images.side'] = side_img.to(device).unsqueeze(0)
        
        # Task description - use task_description mapping like in eval_policy.py
        from src.make_sim_dataset import task_description
        batch['task'] = task_description.get(task, task)
        
        return batch


class SmolVLAGRPOTrainer:
    """
    GRPO trainer for SmolVLA using preference-based optimization.
    More efficient than PPO as it doesn't require value functions.
    """
    def __init__(
        self,
        env: GenesisEnv,
        policy: SmolVLAGRPOPolicy,
        config: Dict,
        device: str = "cuda"
    ):
        self.env = env
        self.policy = policy
        self.config = config
        self.device = device
        
        # Move model to device
        self.policy.to(device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=config['learning_rate']
        )
        
        # Initialize preference buffer
        self.buffer = SmolVLAPreferenceBuffer(
            max_pairs=config['max_preference_pairs']
        )
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize wandb
        wandb.init(
            project=config.get('wandb_project', 'smolvla-grpo'),
            name=config.get('wandb_run_name', f"grpo_smolvla_{config['task']}"),
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
    
    def collect_trajectories(self) -> List[Trajectory]:
        """Collect trajectories for preference learning."""
        trajectories = []
        
        for _ in range(self.config['trajectories_per_update']):
            trajectory = self._collect_single_trajectory()
            if trajectory:
                trajectories.append(trajectory)
        
        return trajectories
    
    def _collect_single_trajectory(self) -> Optional[Trajectory]:
        """Collect a single trajectory."""
        # Reset policy like in eval_policy.py
        self.policy.smolvla_policy.reset()
        
        reset_result = self.env.reset()
        # Handle both tuple and dict returns from env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        done = False
        
        observations = []
        actions = []
        rewards = []
        
        # Video recording setup
        if self.record_video and self.epoch % self.video_freq == 0:
            frame = self._render_combined_frame(obs)
            if frame is not None:
                self.video_frames.append(frame)
        
        step = 0
        max_steps = self.config.get('max_episode_steps', 100)
        
        while not done and step < max_steps:
            # Deep copy observation dictionary
            if isinstance(obs, dict):
                obs_copy = {}
                for k, v in obs.items():
                    if isinstance(v, np.ndarray):
                        # Make a copy to handle negative strides
                        obs_copy[k] = v.copy()
                    elif hasattr(v, 'copy'):
                        obs_copy[k] = v.copy()
                    else:
                        obs_copy[k] = v
            else:
                obs_copy = obs
            observations.append(obs_copy)
            
            # Prepare batch for policy
            batch = self.policy._prepare_batch_for_policy(obs, self.config['task'])
            
            # Debug: Print image shapes
            # if step == 0:  # Only print on first step to avoid spam
            #     for key, value in obs.items():
            #         if 'images' in key and isinstance(value, np.ndarray):
            #             self.logger.info(f"Observation {key} shape: {value.shape}, dtype: {value.dtype}")
            
            # Sample action
            action = self.policy.sample_action(batch, deterministic=False)
            
            # Convert to numpy if tensor and remove batch dimension
            if isinstance(action, torch.Tensor):
                action = action.squeeze(0).cpu().numpy()
            
            # Debug: Print action shape on first step
            # if step == 0:
            #     self.logger.info(f"Action shape: {action.shape}, Action: {action}")
            
            actions.append(action.copy())
            
            # Take step in environment
            step_result = self.env.step(action)
            
            # Handle different step return formats
            if len(step_result) == 4:
                # Old format: (obs, reward, done, info)
                obs, reward, done, info = step_result
            elif len(step_result) == 5:
                # New format: (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raise ValueError(f"Unexpected step result format: {len(step_result)} elements")
            
            rewards.append(reward)
            
            # Record video frame
            if self.record_video and self.epoch % self.video_freq == 0:
                frame = self._render_combined_frame(obs)
                if frame is not None:
                    self.video_frames.append(frame)
            
            step += 1
        
        if len(actions) == 0:
            return None
        
        return Trajectory(
            observations=observations,
            actions=actions,
            rewards=rewards,
            task=self.config['task']
        )
    
    def _render_combined_frame(self, obs: Dict) -> Optional[np.ndarray]:
        """Create combined frame from camera observations."""
        try:
            frames = []
            
            if 'observation.images.front' in obs:
                front_img = obs['observation.images.front']
                if isinstance(front_img, torch.Tensor):
                    front_img = front_img.cpu().numpy()
                # Make a copy to handle negative strides
                if isinstance(front_img, np.ndarray):
                    front_img = front_img.copy()
                # Convert from CHW to HWC if needed
                if front_img.shape[0] == 3:
                    front_img = np.transpose(front_img, (1, 2, 0))
                frames.append(front_img)
            
            if 'observation.images.side' in obs:
                side_img = obs['observation.images.side']
                if isinstance(side_img, torch.Tensor):
                    side_img = side_img.cpu().numpy()
                # Make a copy to handle negative strides
                if isinstance(side_img, np.ndarray):
                    side_img = side_img.copy()
                # Convert from CHW to HWC if needed
                if side_img.shape[0] == 3:
                    side_img = np.transpose(side_img, (1, 2, 0))
                frames.append(side_img)
            
            if frames:
                # Concatenate frames horizontally
                combined = np.concatenate(frames, axis=1)
                # Ensure values are in [0, 255] range
                if combined.max() <= 1.0:
                    combined = (combined * 255).astype(np.uint8)
                return combined
            
        except Exception as e:
            self.logger.warning(f"Failed to render frame: {e}")
        
        return None
    
    def compute_grpo_loss(self, preferred_traj: Trajectory, dispreferred_traj: Trajectory) -> torch.Tensor:
        """
        Compute GRPO loss for a preference pair.
        GRPO loss = -log(sigmoid(log_prob_preferred - log_prob_dispreferred))
        """
        # Compute log probabilities for both trajectories
        log_prob_preferred = self.policy.compute_trajectory_logprob(preferred_traj)
        log_prob_dispreferred = self.policy.compute_trajectory_logprob(dispreferred_traj)
        
        # GRPO loss: maximize difference in log probabilities
        logits = log_prob_preferred - log_prob_dispreferred
        loss = -F.logsigmoid(logits)
        
        return loss
    
    def update_policy(self, trajectories: List[Trajectory]) -> Dict:
        """Update policy using GRPO objective."""
        if not trajectories:
            return {'grpo_loss': 0.0}
        
        # Add trajectories to buffer
        for traj in trajectories:
            self.buffer.add_trajectory(traj)
        
        # Get preference pairs for training
        preference_pairs = self.buffer.get_preference_batch(self.config['batch_size'])
        
        if not preference_pairs:
            return {'grpo_loss': 0.0}
        
        total_loss = 0.0
        num_updates = 0
        
        # Update policy using preference pairs
        for preferred, dispreferred in preference_pairs:
            self.optimizer.zero_grad()
            
            # Compute GRPO loss
            loss = self.compute_grpo_loss(preferred, dispreferred)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('max_grad_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config['max_grad_norm']
                )
            
            # Optimizer step
            self.optimizer.step()
            
            total_loss += loss.item()
            num_updates += 1
        
        avg_loss = total_loss / max(num_updates, 1)
        
        return {'grpo_loss': avg_loss}
    
    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting GRPO training for {self.config['epochs']} epochs")
        
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Collect trajectories
            trajectories = self.collect_trajectories()
            
            if not trajectories:
                self.logger.warning(f"No trajectories collected in epoch {epoch}")
                continue
            
            # Compute metrics
            rewards = [traj.total_reward for traj in trajectories]
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            
            # Update policy
            update_metrics = self.update_policy(trajectories)
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            
            log_data = {
                'epoch': epoch,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'num_trajectories': len(trajectories),
                'epoch_time': epoch_time,
                **update_metrics
            }
            
            wandb.log(log_data)
            
            self.logger.info(
                f"Epoch {epoch}: Reward={mean_reward:.3f}±{std_reward:.3f}, "
                f"Loss={update_metrics.get('grpo_loss', 0):.4f}, Time={epoch_time:.1f}s"
            )
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch)
            
            # Upload video
            if self.record_video and self.video_frames and epoch % self.video_freq == 0:
                self._upload_video(epoch)
                self.video_frames = []
            
            # Clear buffer periodically
            if (epoch + 1) % self.config.get('buffer_clear_freq', 50) == 0:
                self.buffer.clear()
        
        self.logger.info("Training completed!")
        wandb.finish()
    
    def _upload_video(self, epoch: int):
        """Upload video to wandb."""
        try:
            if len(self.video_frames) > 0:
                video_path = f"/tmp/smolvla_epoch_{epoch}.mp4"
                
                # Create video
                with imageio.get_writer(video_path, fps=self.config.get('video_fps', 30)) as writer:
                    for frame in self.video_frames:
                        if frame is not None:
                            writer.append_data(frame)
                
                # Upload to wandb
                wandb.log({f"video_epoch_{epoch}": wandb.Video(video_path)})
                
                # Clean up
                if os.path.exists(video_path):
                    os.remove(video_path)
                    
        except Exception as e:
            self.logger.warning(f"Failed to upload video: {e}")
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config['checkpoint_dir']) / f"epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save policy state dict
        policy_path = checkpoint_dir / "policy.pth"
        torch.save({
            'epoch': epoch,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, policy_path)
        
        # Save SmolVLA policy separately
        smolvla_path = checkpoint_dir / "smolvla_policy"
        self.policy.smolvla_policy.save_pretrained(smolvla_path)
        
        # Upload checkpoint to wandb
        wandb.save(str(policy_path))

def main(config):
    """Main training function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
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
    
    # Create GRPO policy wrapper
    grpo_policy = SmolVLAGRPOPolicy(smolvla_policy)
    
    # Create trainer
    trainer = SmolVLAGRPOTrainer(
        env=env,
        policy=grpo_policy,
        config=config,
        device=device
    )
    
    # Start training
    trainer.train()
    
    # Clean up
    env.close()


if __name__ == "__main__":
    # Training configuration
    config = {
        # Environment
        'task': 'test',
        'observation_height': 480,
        'observation_width': 640,
        'show_viewer': False,
        
        # GRPO hyperparameters
        'epochs': 1000,
        'trajectories_per_update': 5,  # Collect 5 trajectories per update
        'max_preference_pairs': 200,   # Keep max 200 preference pairs
        'learning_rate': 1e-5,         # Lower LR for SmolVLA fine-tuning
        'batch_size': 16,              # Batch size for preference pairs
        'max_episode_steps': 100,      # Max steps per episode
        'max_grad_norm': 0.5,          # Gradient clipping
        
        # Logging and saving
        'wandb_project': 'smolvla-grpo',
        'wandb_run_name': None,  # Will be auto-generated
        'checkpoint_dir': 'outputs/rl_checkpoints2',
        'save_freq': 20,
        'buffer_clear_freq': 50,  # Clear buffer every N epochs
        
        # Video recording
        'record_video': True,
        'video_freq': 10,  # Record video every 10 epochs
        'video_fps': 30,
        
        # Model
        'pretrained_model_path': "outputs/train/smolvla_test_0/checkpoints/last/pretrained_model",
    }
    main(config)

# uv run -m src.rl_smolvla