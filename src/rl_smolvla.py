"""PPO training for SmolVLA policy."""

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
import torch.nn as nn
import torch.nn.functional as F
import wandb
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.genesis_env import GenesisEnv
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from src.make_sim_dataset import task_description

@dataclass
class ActionSequence:
    """Single action sequence for PPO training"""
    observation: Dict
    action: np.ndarray  # n_action_steps分のアクション
    reward: float  # sequence全体の報酬をn_action_stepsで割った値
    value: float = 0.0  # 価値関数の推定値
    log_prob: Optional[float] = None
    task: str = ""

class ValueNetwork(nn.Module):
    """価値関数ネットワーク"""
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


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
    
    def compute_action_logprob(self, obs: Dict, action: np.ndarray, task: str) -> torch.Tensor:
        """単一のaction sequenceのlog probabilityを計算"""
        self.train()
        batch = self._prepare_batch(obs, task)
        action_tensor = torch.from_numpy(action).float().to(next(self.parameters()).device)
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
        batch['actions'] = action_tensor
        
        try:
            outputs = self.smolvla_policy.forward(batch)
            log_prob = -outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0, requires_grad=True)
            return log_prob
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to compute log prob: {e}")
            return torch.tensor(-1.0, requires_grad=True)
    
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

class PPOTrainer:
    def __init__(self, env: GenesisEnv, policy: SmolVLAPolicyWrapper, config: Dict, device: str = "cuda"):
        self.env = env
        self.policy = policy.to(device)
        self.config = config
        self.device = device
        
        # 価値関数ネットワーク
        self.value_network = ValueNetwork(
            state_dim=config.get('state_dim', 7),  # agent_posの次元
            hidden_dim=config.get('value_hidden_dim', 256)
        ).to(device)
        
        # オプティマイザー
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=config['policy_lr']
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(), 
            lr=config['value_lr']
        )
        
        self.logger = logging.getLogger(__name__)
        self.epoch = 0
        self.video_frames = []
        
        # 価値関数安定化用の変数
        self.value_loss_history = []
        self.value_stable_threshold = config.get('value_stable_threshold', 0.01)
        self.value_stable_window = config.get('value_stable_window', 10)
        self.value_update_epochs = config.get('value_update_epochs', 5)
        
        wandb.init(
            project=config.get('wandb_project', 'smolvla-ppo'),
            name=config.get('wandb_run_name', f"ppo_{config['task']}"),
            config=config,
            sync_tensorboard=False
        )
    
    def collect_trajectories(self) -> List[ActionSequence]:
        """batch_size分のエピソードを実行してaction sequenceを収集"""
        all_sequences = []
        
        for episode_idx in range(self.config['batch_size']):
            sequences = self._collect_episode_sequences()
            if sequences:
                all_sequences.extend(sequences)
                self.logger.info(f"Episode {episode_idx + 1}: collected {len(sequences)} sequences")
        
        return all_sequences
    
    def _collect_episode_sequences(self) -> List[ActionSequence]:
        """1エピソードを実行してaction sequenceを収集"""
        self.policy.smolvla_policy.reset()
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs
        
        sequences = []
        done = False
        step = 0
        max_steps = self.config.get('max_episode_steps', 100)
        n_action_steps = self.config['n_action_steps']
        
        while not done and step < max_steps:
            obs_copy = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
            
            # action sequenceを生成
            batch = self.policy._prepare_batch(obs, self.config['task'])
            action_sequence = self.policy.sample_action(batch)
            
            if isinstance(action_sequence, torch.Tensor):
                action_sequence = action_sequence.cpu().numpy()
            
            # 価値関数による状態価値推定
            state_tensor = self._extract_state_tensor(obs)
            with torch.no_grad():
                value = self.value_network(state_tensor).item()
            
            # Record video frame
            if self.config.get('record_video', False) and self.epoch % self.config.get('video_freq', 10) == 0:
                frame = self._render_frame(obs)
                if frame is not None:
                    self.video_frames.append(frame)
            
            # n_action_steps分のアクションを実行して報酬を収集
            sequence_rewards = []
            for action_idx in range(min(n_action_steps, len(action_sequence))):
                single_action = action_sequence[action_idx]
                obs, reward, done, truncated, info = self.env.step(single_action)
                sequence_rewards.append(reward)
                step += 1
                
                if done or truncated:
                    break
            
            # sequence全体の報酬をn_action_stepsで割る
            avg_reward = sum(sequence_rewards) / len(sequence_rewards) if sequence_rewards else 0.0
            
            # ActionSequence作成
            action_seq = ActionSequence(
                observation=obs_copy,
                action=action_sequence,
                reward=avg_reward,
                value=value,
                task=task_description.get(self.config['task'], self.config['task'])
            )
            sequences.append(action_seq)
            
            if done:
                break
        
        return sequences
    
    def _extract_state_tensor(self, obs: Dict) -> torch.Tensor:
        """観測から状態テンソルを抽出"""
        if 'agent_pos' in obs:
            state = torch.from_numpy(obs['agent_pos'].copy()).float().to(self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            return state
        else:
            # デフォルト状態
            return torch.zeros(1, self.config.get('state_dim', 7)).to(self.device)
    
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
    
    def compute_advantages(self, sequences: List[ActionSequence]) -> Tuple[np.ndarray, np.ndarray]:
        """GAEを使ってアドバンテージを計算"""
        rewards = np.array([seq.reward for seq in sequences])
        values = np.array([seq.value for seq in sequences])
        
        # 最後の状態の価値（終了状態なので0）
        next_values = np.append(values[1:], 0.0)
        
        # TD誤差
        deltas = rewards + self.config['gamma'] * next_values - values
        
        # GAE
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.config['gamma'] * self.config['gae_lambda'] * gae
            advantages[t] = gae
        
        # リターン
        returns = advantages + values
        
        return advantages, returns
    
    def is_value_function_stable(self) -> bool:
        """価値関数が安定しているかを判定"""
        if len(self.value_loss_history) < self.value_stable_window:
            return False
        
        recent_losses = self.value_loss_history[-self.value_stable_window:]
        loss_std = np.std(recent_losses)
        return loss_std < self.value_stable_threshold
    
    def update_value_function(self, sequences: List[ActionSequence], returns: np.ndarray) -> float:
        """価値関数を更新"""
        total_value_loss = 0.0
        
        for _ in range(self.value_update_epochs):
            states = []
            for seq in sequences:
                state = self._extract_state_tensor(seq.observation)
                states.append(state)
            
            if not states:
                continue
                
            states_tensor = torch.cat(states, dim=0)
            returns_tensor = torch.from_numpy(returns).float().to(self.device)
            
            # 価値関数の予測
            predicted_values = self.value_network(states_tensor)
            
            # 価値関数のloss
            value_loss = F.mse_loss(predicted_values, returns_tensor)
            
            # 価値関数の更新
            self.value_optimizer.zero_grad()
            value_loss.backward()
            if self.config.get('max_grad_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.value_network.parameters(), 
                    self.config['max_grad_norm']
                )
            self.value_optimizer.step()
            
            total_value_loss += value_loss.item()
        
        avg_value_loss = total_value_loss / self.value_update_epochs
        self.value_loss_history.append(avg_value_loss)
        
        # 履歴を適切なサイズに保つ
        if len(self.value_loss_history) > self.value_stable_window * 2:
            self.value_loss_history = self.value_loss_history[-self.value_stable_window:]
        
        return avg_value_loss
    
    def update_policy(self, sequences: List[ActionSequence], advantages: np.ndarray) -> Dict:
        """PPOでポリシーを更新"""
        if not sequences:
            return {'policy_loss': 0.0, 'num_sequences': 0}
        
        # アドバンテージの正規化
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 古いlog probを計算
        old_log_probs = []
        for seq in sequences:
            log_prob = self.policy.compute_action_logprob(
                seq.observation, seq.action, seq.task
            )
            old_log_probs.append(log_prob.detach())
        
        old_log_probs_tensor = torch.stack(old_log_probs)
        advantages_tensor = torch.from_numpy(advantages).float().to(self.device)
        
        total_policy_loss = 0.0
        
        for ppo_epoch in range(self.config.get('ppo_epochs', 4)):
            # 新しいlog probを計算
            new_log_probs = []
            for seq in sequences:
                log_prob = self.policy.compute_action_logprob(
                    seq.observation, seq.action, seq.task
                )
                new_log_probs.append(log_prob)
            
            new_log_probs_tensor = torch.stack(new_log_probs)
            
            # 重要度サンプリング比
            ratio = torch.exp(new_log_probs_tensor - old_log_probs_tensor)
            
            # PPO loss
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(
                ratio, 
                1.0 - self.config['clip_epsilon'], 
                1.0 + self.config['clip_epsilon']
            ) * advantages_tensor
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # ポリシーの更新
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            if self.config.get('max_grad_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    self.config['max_grad_norm']
                )
            self.policy_optimizer.step()
            
            total_policy_loss += policy_loss.item()
        
        avg_policy_loss = total_policy_loss / self.config.get('ppo_epochs', 4)
        
        return {
            'policy_loss': avg_policy_loss,
            'num_sequences': len(sequences),
            'avg_reward': np.mean([seq.reward for seq in sequences]),
            'avg_advantage': np.mean(advantages)
        }
    
    def train(self):
        self.logger.info(f"Starting PPO training for {self.config['epochs']} epochs")
        
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            
            # データ収集
            sequences = self.collect_trajectories()
            if not sequences:
                self.logger.warning("No sequences collected, skipping epoch")
                continue
            
            # アドバンテージ計算
            advantages, returns = self.compute_advantages(sequences)
            
            # 価値関数の更新
            value_loss = self.update_value_function(sequences, returns)
            
            # 価値関数が安定している場合のみポリシーを更新
            policy_update_info = {'policy_loss': 0.0, 'policy_updated': False}
            if self.is_value_function_stable():
                policy_update_info = self.update_policy(sequences, advantages)
                policy_update_info['policy_updated'] = True
                self.logger.info("Policy updated (value function is stable)")
            else:
                self.logger.info(f"Policy update skipped (value function not stable, loss std: {np.std(self.value_loss_history[-self.value_stable_window:]) if len(self.value_loss_history) >= self.value_stable_window else 'N/A'})")
            
            # ログ記録
            wandb.log({
                'epoch': epoch,
                'value_loss': value_loss,
                'policy_loss': policy_update_info['policy_loss'],
                'avg_reward': np.mean([seq.reward for seq in sequences]),
                'num_sequences': len(sequences),
                'value_function_stable': self.is_value_function_stable(),
                'policy_updated': policy_update_info['policy_updated']
            })
            
            # ビデオ保存
            if (self.config.get('record_video', False) and 
                epoch % self.config.get('video_freq', 10) == 0 and 
                self.video_frames):
                self._upload_video(epoch)
                self.video_frames = []
            
            # チェックポイント保存
            if epoch % self.config.get('save_freq', 50) == 0:
                self.save_checkpoint(epoch)
        
        wandb.finish()
    
    def _upload_video(self, epoch: int):
        try:
            video_array = np.array(self.video_frames)
            wandb.log({
                f"video_epoch_{epoch}": wandb.Video(
                    video_array, 
                    fps=self.config.get('video_fps', 30), 
                    format="mp4"
                )
            })
        except Exception as e:
            self.logger.warning(f"Failed to upload video: {e}")
    
    def save_checkpoint(self, epoch: int):
        checkpoint_dir = Path(self.config['checkpoint_dir']) / f"epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'policy_state_dict': self.policy.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'value_loss_history': self.value_loss_history,
            'config': self.config
        }
        
        policy_path = checkpoint_dir / "checkpoint.pth"
        torch.save(checkpoint, policy_path)
        
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
    trainer = PPOTrainer(env, policy, config, device)
    trainer.train()
    env.close()

if __name__ == "__main__":
    config = {
        'task': 'simple_pick',
        'observation_height': 512,
        'observation_width': 512,
        'show_viewer': False,
        'epochs': 1000,
        'batch_size': 8,  # 並行して実行するエピソード数
        'policy_lr': 1e-5,
        'value_lr': 1e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'ppo_epochs': 4,
        'max_episode_steps': 500,
        'max_grad_norm': 0.5,
        'wandb_project': 'smolvla-ppo',
        'wandb_run_name': None,
        'checkpoint_dir': 'outputs/rl_checkpoints_ppo',
        'save_freq': 50,
        'record_video': True,
        'video_freq': 10,
        'video_fps': 30,
        'pretrained_model_path': "outputs/train/smolvla_test_0/checkpoints/last/pretrained_model",
        'n_action_steps': 50,
        'state_dim': 7,  # agent_posの次元
        'value_hidden_dim': 256,
        'value_stable_threshold': 0.01,  # 価値関数の安定性判定閾値
        'value_stable_window': 10,       # 安定性判定のためのウィンドウサイズ
        'value_update_epochs': 5,        # 価値関数の更新エポック数
    }
    main(config)