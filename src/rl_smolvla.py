"""GRPO training for SmolVLA policy."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import imageio
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.genesis_env import GenesisEnv
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from src.make_sim_dataset import task_description

@dataclass
class Trajectory:
    observations: List[Dict]
    actions: List[np.ndarray]
    rewards: List[float] 
    task: str
    total_reward: float = 0.0
    
    def __post_init__(self):
        self.total_reward = sum(self.rewards)

class PreferenceBuffer:
    def __init__(self, max_pairs: int = 1000):
        self.max_pairs = max_pairs
        self.trajectories = []
        self.preference_pairs = []
        
    def add_trajectory(self, trajectory: Trajectory):
        self.trajectories.append(trajectory)
        if len(self.trajectories) >= 2:
            self._create_preference_pairs()
        if len(self.trajectories) > self.max_pairs * 2:
            self.trajectories = self.trajectories[-self.max_pairs:]
    
    def _create_preference_pairs(self):
        if len(self.trajectories) < 2:
            return
        recent = self.trajectories[-10:]
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                t1, t2 = recent[i], recent[j]
                preferred, dispreferred = (t1, t2) if t1.total_reward > t2.total_reward else (t2, t1)
                self.preference_pairs.append((preferred, dispreferred))
        if len(self.preference_pairs) > self.max_pairs:
            self.preference_pairs = self.preference_pairs[-self.max_pairs:]
    
    def get_preference_batch(self, batch_size: int) -> List[Tuple[Trajectory, Trajectory]]:
        return random.sample(self.preference_pairs, min(batch_size, len(self.preference_pairs)))
    
    def clear(self):
        self.trajectories.clear()
        self.preference_pairs.clear()


class SmolVLAPolicyWrapper:
    def __init__(self, smolvla_policy: SmolVLAPolicy):
        self.smolvla_policy = smolvla_policy
        
    def to(self, device):
        self.smolvla_policy.to(device)
        return self
    
    def parameters(self):
        return self.smolvla_policy.parameters()
    
    def state_dict(self):
        return self.smolvla_policy.state_dict()
    
    def load_state_dict(self, state_dict):
        self.smolvla_policy.load_state_dict(state_dict)
    
    def train(self):
        self.smolvla_policy.train()
    
    def eval(self):
        self.smolvla_policy.eval()
    
    def sample_action(self, batch: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            action = self.smolvla_policy.select_action(batch)
        return action
    
    def compute_trajectory_logprob(self, trajectory: Trajectory) -> torch.Tensor:
        self.train()
        total_logprob = 0.0
        for obs, action in zip(trajectory.observations, trajectory.actions):
            batch = self._prepare_batch(obs, trajectory.task)
            action_tensor = torch.from_numpy(action).float().to(next(self.parameters()).device)
            if action_tensor.dim() == 1:
                action_tensor = action_tensor.unsqueeze(0)
            batch['actions'] = action_tensor
            try:
                outputs = self.smolvla_policy.forward(batch)
                step_logprob = -outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0, requires_grad=True)
                total_logprob += step_logprob
            except:
                total_logprob += torch.tensor(-1.0, requires_grad=True)
        return total_logprob
    
    def _prepare_batch(self, obs: Dict, task: str) -> Dict[str, torch.Tensor]:
        batch = {}
        device = next(self.parameters()).device
        
        if 'agent_pos' in obs:
            agent_pos = torch.from_numpy(obs['agent_pos'].copy()).float().to(device)
            if agent_pos.dim() == 1:
                agent_pos = agent_pos.unsqueeze(0)
            batch['observation.state'] = agent_pos
        
        for key in ['observation.images.front', 'observation.images.side']:
            if key in obs:
                img = obs[key].copy()
                img = torch.from_numpy(img).float() / 255.0
                if img.ndim == 3 and img.shape[2] in [1, 3, 4]:
                    img = img.permute(2, 0, 1)
                elif img.ndim == 2:
                    img = img.unsqueeze(0)
                batch[key] = img.to(device).unsqueeze(0)
        
        batch['task'] = task_description.get(task, task)
        return batch

class GRPOTrainer:
    def __init__(self, env: GenesisEnv, policy: SmolVLAPolicyWrapper, config: Dict, device: str = "cuda"):
        self.env = env
        self.policy = policy.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['learning_rate'])
        self.buffer = PreferenceBuffer(config['max_preference_pairs'])
        self.logger = logging.getLogger(__name__)
        self.epoch = 0
        self.video_frames = []
        
        wandb.init(
            project=config.get('wandb_project', 'smolvla-grpo'),
            name=config.get('wandb_run_name', f"grpo_{config['task']}"),
            config=config,
            sync_tensorboard=False
        )
    
    def collect_trajectories(self) -> List[Trajectory]:
        trajectories = []
        for _ in range(self.config['trajectories_per_update']):
            trajectory = self._collect_trajectory()
            if trajectory:
                trajectories.append(trajectory)
        return trajectories
    
    def _collect_trajectory(self) -> Optional[Trajectory]:
        self.policy.smolvla_policy.reset()
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs
        
        observations, actions, rewards = [], [], []
        done = False
        step = 0
        max_steps = self.config.get('max_episode_steps', 100)
        
        while not done and step < max_steps:
            obs_copy = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
            observations.append(obs_copy)
            
            batch = self.policy._prepare_batch(obs, self.config['task'])
            action = self.policy.sample_action(batch)
            
            if isinstance(action, torch.Tensor):
                action = action.squeeze(0).cpu().numpy()
            actions.append(action.copy())
            
            # Record video frame
            if self.config.get('record_video', False) and self.epoch % self.config.get('video_freq', 10) == 0:
                frame = self._render_frame(obs)
                if frame is not None:
                    self.video_frames.append(frame)
            
            step_result = self.env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            rewards.append(reward)
            step += 1
        
        return Trajectory(observations, actions, rewards, self.config['task']) if actions else None
    
    def _render_frame(self, obs: Dict) -> Optional[np.ndarray]:
        try:
            frames = []
            for key in ['observation.images.front', 'observation.images.side']:
                if key in obs:
                    img = obs[key]
                    if isinstance(img, torch.Tensor):
                        img = img.cpu().numpy()
                    if isinstance(img, np.ndarray):
                        img = img.copy()
                    if img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))
                    frames.append(img)
            
            if frames:
                combined = np.concatenate(frames, axis=1)
                if combined.max() <= 1.0:
                    combined = (combined * 255).astype(np.uint8)
                return combined
        except Exception as e:
            self.logger.warning(f"Failed to render frame: {e}")
        return None
    
    def compute_grpo_loss(self, preferred: Trajectory, dispreferred: Trajectory) -> torch.Tensor:
        log_prob_preferred = self.policy.compute_trajectory_logprob(preferred)
        log_prob_dispreferred = self.policy.compute_trajectory_logprob(dispreferred)
        logits = log_prob_preferred - log_prob_dispreferred
        return -F.logsigmoid(logits)
    
    def update_policy(self, trajectories: List[Trajectory]) -> Dict:
        if not trajectories:
            return {'grpo_loss': 0.0}
        
        for traj in trajectories:
            self.buffer.add_trajectory(traj)
        
        preference_pairs = self.buffer.get_preference_batch(self.config['batch_size'])
        if not preference_pairs:
            return {'grpo_loss': 0.0}
        
        total_loss = 0.0
        for preferred, dispreferred in preference_pairs:
            self.optimizer.zero_grad()
            loss = self.compute_grpo_loss(preferred, dispreferred)
            loss.backward()
            
            if self.config.get('max_grad_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['max_grad_norm'])
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return {'grpo_loss': total_loss / len(preference_pairs)}
    
    def train(self):
        self.logger.info(f"Starting GRPO training for {self.config['epochs']} epochs")
        
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            start_time = time.time()
            
            trajectories = self.collect_trajectories()
            if not trajectories:
                continue
            
            rewards = [traj.total_reward for traj in trajectories]
            update_metrics = self.update_policy(trajectories)
            
            log_data = {
                'epoch': epoch,
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'epoch_time': time.time() - start_time,
                **update_metrics
            }
            
            wandb.log(log_data)
            self.logger.info(f"Epoch {epoch}: Reward={np.mean(rewards):.3f}±{np.std(rewards):.3f}")
            
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch)
            
            if self.video_frames and epoch % self.config.get('video_freq', 10) == 0:
                self._upload_video(epoch)
                self.video_frames = []
            
            if (epoch + 1) % self.config.get('buffer_clear_freq', 50) == 0:
                self.buffer.clear()
        
        wandb.finish()
    
    def _upload_video(self, epoch: int):
        try:
            if self.video_frames:
                video_path = f"/tmp/smolvla_epoch_{epoch}.mp4"
                with imageio.get_writer(video_path, fps=30) as writer:
                    for frame in self.video_frames:
                        if frame is not None:
                            writer.append_data(frame)
                wandb.log({f"video_epoch_{epoch}": wandb.Video(video_path)})
                if os.path.exists(video_path):
                    os.remove(video_path)
        except Exception as e:
            self.logger.warning(f"Failed to upload video: {e}")
    
    def save_checkpoint(self, epoch: int):
        checkpoint_dir = Path(self.config['checkpoint_dir']) / f"epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        policy_path = checkpoint_dir / "policy.pth"
        torch.save({
            'epoch': epoch,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, policy_path)
        
        smolvla_path = checkpoint_dir / "smolvla_policy"
        self.policy.smolvla_policy.save_pretrained(smolvla_path)
        wandb.save(str(policy_path))

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
    
    policy = SmolVLAPolicyWrapper(smolvla_policy)
    trainer = GRPOTrainer(env, policy, config, device)
    trainer.train()
    env.close()

if __name__ == "__main__":
    config = {
        'task': 'simple_pick',
        'observation_height': 512,
        'observation_width': 512,
        'show_viewer': False,
        'epochs': 1000,
        'trajectories_per_update': 5,
        'max_preference_pairs': 200,
        'learning_rate': 1e-5,
        'batch_size': 16,
        'max_episode_steps': 500, # エピソードの最大ステップ数
        'max_grad_norm': 0.5,
        'wandb_project': 'smolvla-grpo',
        'wandb_run_name': None,
        'checkpoint_dir': 'outputs/rl_checkpoints2',
        'save_freq': 20,
        'buffer_clear_freq': 50,
        'record_video': True,
        'video_freq': 10,
        'video_fps': 30,
        'pretrained_model_path': "outputs/train/smolvla_test_0/checkpoints/last/pretrained_model",
        'n_action_steps': 50, # action chunkのうち，何ステップを実行するか
    }
    main(config)
    
# 動画の保存方法が良くない。それぞれ名前を変えない。