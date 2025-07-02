import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.distributions as dist
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.constants import ACTION
import numpy as np
import wandb
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from env.genesis_env import GenesisEnv
from pathlib import Path
import cv2

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

class ImageValueNetwork(nn.Module):
    """画像特徴量を使った価値関数ネットワーク"""
    def __init__(self, smolvla_policy: SmolVLAPolicy, hidden_dim: int = 256):
        super().__init__()
        self.smolvla_policy = smolvla_policy
        
        # SmolVLAのビジョンエンコーダーの実際の出力次元を動的に取得
        self._vision_feature_dim = None
        self.hidden_dim = hidden_dim
        
        # 価値関数のヘッドは遅延初期化
        self.value_head = None
    
    def forward(self, obs_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """観測から価値を予測"""
        with torch.no_grad():
            # SmolVLAのビジョンエンコーダーを使って特徴量を抽出
            vision_features = self._extract_vision_features(obs_batch)
        
        # 価値関数のヘッドを遅延初期化
        if self.value_head is None:
            self._vision_feature_dim = vision_features.shape[-1]
            self.value_head = nn.Sequential(
                nn.Linear(self._vision_feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(), 
                nn.Linear(self.hidden_dim, 1)
            ).to(vision_features.device)
        
        value = self.value_head(vision_features)
        return value.squeeze(-1)
    
    def _extract_vision_features(self, obs_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """SmolVLAのビジョンエンコーダーから特徴量を抽出"""
        # 画像を取得
        images = []
        img_masks = []
        
        for key in ['observation.images.front', 'observation.images.side']:
            if key in obs_batch:
                img = obs_batch[key]
                if img.dim() == 4:  # [B, C, H, W]
                    images.append(img)
                    batch_size = img.shape[0]
                    device = img.device
                    img_masks.append(torch.ones(batch_size, dtype=torch.bool, device=device))
        
        if not images:
            # デフォルトの特徴量を返す（float32型で）
            # 次元は実際のビジョン特徴量と同じにする必要があるため、
            # 一度実際の画像で次元を確認してから設定する
            batch_size = obs_batch[list(obs_batch.keys())[0]].shape[0]
            device = obs_batch[list(obs_batch.keys())[0]].device
            # デフォルトは960次元（SmolVLAの実際の出力次元）
            default_dim = getattr(self, '_vision_feature_dim', 960)
            return torch.zeros(batch_size, default_dim, dtype=torch.float32, device=device)
        
        # SmolVLAのビジョンエンコーダーを使用して特徴量を抽出
        # VLMWithExpertモデルのembed_imageメソッドを使用
        all_features = []
        for img in images:
            # 画像を[-1, 1]の範囲に正規化（SmolVLAの要求に合わせる）
            if img.max() <= 1.0:
                img = img * 2.0 - 1.0
            
            # 画像をエンベッド
            img_features = self.smolvla_policy.model.vlm_with_expert.embed_image(img)
            
            # データ型をfloat32に変換
            img_features = img_features.float()
            
            # グローバル平均プールを適用して固定次元に
            if img_features.dim() > 2:
                pooled_features = img_features.mean(dim=1)  # [B, seq_len, dim] -> [B, dim]
            else:
                pooled_features = img_features
            
            all_features.append(pooled_features)
        
        # 複数画像がある場合は平均
        if len(all_features) > 1:
            vision_features = torch.stack(all_features, dim=0).mean(dim=0)
        else:
            vision_features = all_features[0]
        
        # 最終的にfloat32型であることを確認
        return vision_features.float()


class SmolVLAPolicyWrapper(nn.Module):
    def __init__(self, smolvla_policy: SmolVLAPolicy, action_dim: int, initial_std: float = 0.1):
        super().__init__()
        self.smolvla_policy = smolvla_policy
        # PPO用の標準偏差パラメータを追加
        self.log_std = nn.Parameter(torch.full((action_dim,), np.log(initial_std)))
        
    def parameters(self):
        # SmolVLAのパラメータとlog_stdの両方を返す
        return list(self.smolvla_policy.parameters()) + [self.log_std]
    
    def state_dict(self):
        return {
            'smolvla_policy': self.smolvla_policy.state_dict(),
            'log_std': self.log_std
        }
    
    def load_state_dict(self, state_dict):
        if 'smolvla_policy' in state_dict:
            self.smolvla_policy.load_state_dict(state_dict['smolvla_policy'])
        if 'log_std' in state_dict:
            self.log_std.data = state_dict['log_std']
    
    def train(self):
        super().train()
        self.smolvla_policy.train()
    
    def eval(self):
        super().eval()
        self.smolvla_policy.eval()
    
    def sample_action(self, batch: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """確率的行動選択を実装"""
        # SmolVLAから決定論的な平均行動を取得
        with torch.no_grad():
            mean_action = self.smolvla_policy.select_action(batch)
        
        if deterministic:
            return mean_action
        
        # 確率的サンプリング
        # log_stdの値をクランプして安定化
        self.log_std.data = torch.clamp(self.log_std.data, min=-10.0, max=2.0)
        
        # デバッグ用ログ出力
        logger = logging.getLogger(__name__)
        if torch.isnan(self.log_std).any():
            logger.warning(f"NaN detected in log_std: {self.log_std}")
            self.log_std.data.fill_(np.log(0.1))  # デフォルト値にリセット
        
        std = torch.exp(self.log_std)
        
        # std値のチェック
        if torch.isnan(std).any() or (std <= 0).any():
            logger.warning(f"Invalid std values detected: {std}")
            std = torch.full_like(std, 0.1)  # デフォルト値に置換
        
        logger.debug(f"log_std: {self.log_std.data}, std: {std}")
        
        # mean_actionの形状に合わせてstdを調整
        if mean_action.dim() == 2:  # [n_action_steps, action_dim]
            std = std.unsqueeze(0).expand(mean_action.shape[0], -1)
        
        dist = torch.distributions.Normal(mean_action, std)
        sampled_action = dist.sample()
        
        return sampled_action
    
    def compute_action_logprob(self, obs: Dict, action: np.ndarray, task: str) -> torch.Tensor:
        """ガウス分布ベースのlog probabilityを計算"""
        distribution = self.compute_action_distribution(obs, task)
        
        # actionをtensorに変換
        action_tensor = torch.from_numpy(action).float().to(self.log_std.device)
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
        
        # ガウス分布のlog probabilityを計算
        log_prob = distribution.log_prob(action_tensor)
        
        # 数値的安定性のために平均を取る（全次元の和ではなく）
        log_prob = log_prob.mean()  # 次元数で正規化
        
        return log_prob
    
    def compute_action_distribution(self, obs: Dict, task: str) -> dist.Normal:
        """ガウス分布オブジェクトを返す"""
        batch = self._prepare_batch(obs, task)
        
        # SmolVLAから決定論的な平均行動を取得
        with torch.no_grad():
            mean_action = self.smolvla_policy.select_action(batch)
        
        # log_stdの値をクランプして安定化
        self.log_std.data = torch.clamp(self.log_std.data, min=-10.0, max=2.0)
        
        # デバッグ用ログ出力
        logger = logging.getLogger(__name__)
        if torch.isnan(self.log_std).any():
            logger.warning(f"NaN detected in log_std (compute_action_distribution): {self.log_std}")
            self.log_std.data.fill_(np.log(0.1))  # デフォルト値にリセット
        
        # 標準偏差を取得
        std = torch.exp(self.log_std)
        
        # std値のチェック
        if torch.isnan(std).any() or (std <= 0).any():
            logger.warning(f"Invalid std values detected (compute_action_distribution): {std}")
            std = torch.full_like(std, 0.1)  # デフォルト値に置換
        
        if mean_action.dim() == 2:  # [n_action_steps, action_dim]
            std = std.unsqueeze(0).expand(mean_action.shape[0], -1)
        
        # ガウス分布を作成して返す
        return dist.Normal(mean_action, std)
    
    def _prepare_batch(self, obs: Dict, task_desc: str) -> Dict[str, torch.Tensor]:
        batch = {}
        device = self.log_std.device
        
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
        
        batch['task'] = task_desc
        return batch

class PPOTrainer:
    def __init__(self, env: GenesisEnv, policy: SmolVLAPolicyWrapper, config: Dict, device: str = "cuda"):
        self.env = env
        self.policy = policy.to(device)
        self.config = config
        self.device = device
        
        # 価値関数ネットワーク（画像特徴量を使用）
        if config.get('use_image_value_network', True):
            self.value_network = ImageValueNetwork(
                smolvla_policy=policy.smolvla_policy,
                hidden_dim=config.get('value_hidden_dim', 256)
            ).to(device)
            self.use_image_value = True
        else:
            self.value_network = ValueNetwork(
                state_dim=config.get('state_dim', 7),
                hidden_dim=config.get('value_hidden_dim', 256)
            ).to(device)
            self.use_image_value = False
        
        # オプティマイザー（分離型）
        # PPO用のlog_stdのみを学習
        self.ppo_optimizer = torch.optim.Adam(
            [self.policy.log_std], 
            lr=config['policy_lr']
        )
        # SmolVLA全体を低い学習率で学習
        self.smolvla_optimizer = torch.optim.Adam(
            self.policy.smolvla_policy.parameters(),
            lr=config.get('smolvla_lr', 1e-6)
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
        
        # SmolVLA学習制御用の変数
        self.smolvla_warmup_epochs = config.get('smolvla_warmup_epochs', 50)
        self.flow_matching_coef = config.get('flow_matching_coef', 0.1)
        
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
                # self.logger.info(f"Episode {episode_idx + 1}: collected {len(sequences)} sequences")
        
        return all_sequences
    
    def _collect_episode_sequences(self) -> List[ActionSequence]:
        """1エピソードを実行してaction chunkを収集（1 chunk = 1 PPO action）"""
        self.policy.smolvla_policy.reset()
        obs = self.env.reset()
        task_desc = self.env.get_task_description()
        if isinstance(obs, tuple):
            obs, _ = obs
        
        sequences = []
        done = False
        step_count = 0
        max_steps = self.config.get('max_episode_steps', 100)
        n_action_steps = self.config['n_action_steps']
        
        while not done and step_count < max_steps:
            obs_copy = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
            # action chunk全体を実行して報酬を累積
            chunk_total_reward = 0.0
            action_chunk = []
            # 1つのaction chunkを生成（確率的サンプリング）
            batch = self.policy._prepare_batch(obs, self.env.get_task_description())
            for _ in range(n_action_steps):
                action = self.policy.sample_action(batch, deterministic=False)[0]
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                action_chunk.append(action)
                obs, reward, done, truncated, info = self.env.step(action)
                # Record video frame
                if self.config.get('record_video', False) and self.epoch % self.config.get('video_freq', 10) == 0:
                    frame = self._render_frame(obs, task_desc)
                    if frame is not None:
                        self.video_frames.append(frame)
                chunk_total_reward += reward
                step_count += 1
                if done or truncated:
                    break
            # 価値関数による状態価値推定
            if self.use_image_value:
                obs_batch = self.policy._prepare_batch(obs_copy, task_desc)
                with torch.no_grad():
                    value = self.value_network(obs_batch).item()
            else:
                state_tensor = self._extract_state_tensor(obs)
                with torch.no_grad():
                    value = self.value_network(state_tensor).item()
            # ActionSequence作成（1 chunk = 1 action）
            action_seq = ActionSequence(
                observation=obs_copy,
                action=np.array(action_chunk),  # n_action_steps分のアクション
                reward=chunk_total_reward / n_action_steps,  # chunk全体の累積報酬を正規化
                value=value,
                task=task_desc
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
    
    def _render_frame(self, obs: Dict, task_desc=None) -> Optional[np.ndarray]:
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
                if task_desc:
                    # cv2.putText(combined, task_desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    # 文字を左下に配置
                    cv2.putText(combined, task_desc, (10, combined.shape[0] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                (255, 255, 255), 2)
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
            if self.use_image_value:
                # 画像価値関数の場合
                predicted_values = []
                for seq in sequences:
                    obs_batch = self.policy._prepare_batch(seq.observation, seq.task)
                    value = self.value_network(obs_batch)
                    predicted_values.append(value)
                
                if not predicted_values:
                    continue
                    
                predicted_values_tensor = torch.stack(predicted_values).squeeze(-1)
                returns_tensor = torch.from_numpy(returns).float().to(self.device)
                
                # 価値関数のloss
                value_loss = F.mse_loss(predicted_values_tensor, returns_tensor)
            else:
                # 従来の状態価値関数の場合
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
                
                # テンソル形状を合わせる
                if predicted_values.dim() > 1:
                    predicted_values = predicted_values.squeeze(-1)
                
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
    
    def compute_flow_matching_loss(self, sequences: List[ActionSequence]) -> torch.Tensor:
        """SmolVLAのFlow Matching損失を計算"""
        if not sequences:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        num_valid = 0
        
        for seq in sequences:
            # バッチを準備
            batch = self.policy._prepare_batch(seq.observation, seq.task)
            
            # アクションをテンソル形式に変換
            action_tensor = torch.from_numpy(seq.action).float().to(self.device).unsqueeze(0)  # [1, n_action_steps, action_dim]
            loss, _ = self.policy.smolvla_policy.forward(
                {**batch, ACTION: action_tensor}
            )
            
            total_loss = total_loss + loss
            num_valid += 1
        
        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def should_update_smolvla(self, epoch: int) -> bool:
        """SmolVLAの更新タイミングを制御"""
        return (self.is_value_function_stable() and 
                epoch >= self.smolvla_warmup_epochs)
    
    def update_policy_hybrid(self, sequences: List[ActionSequence], advantages: np.ndarray, epoch: int) -> Dict:
        """PPO損失とFlow Matching損失を組み合わせたハイブリッド学習"""
        if not sequences:
            return {'policy_loss': 0.0, 'flow_matching_loss': 0.0, 'num_sequences': 0}
        
        # アドバンテージの正規化
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        advantages_tensor = torch.from_numpy(advantages).float().to(self.device)
        
        # 古いlog probを一括計算
        old_log_probs = []
        for seq in sequences:
            try:
                log_prob = self.policy.compute_action_logprob(
                    seq.observation, seq.action, seq.task
                )
                old_log_probs.append(log_prob.detach())
            except Exception as e:
                self.logger.warning(f"Failed to compute old log prob: {e}")
                old_log_probs.append(torch.tensor(-10.0, device=self.device))
        
        old_log_probs_tensor = torch.stack(old_log_probs)
        
        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        total_kl_div = 0.0
        total_flow_matching_loss = 0.0
        
        for ppo_epoch in range(self.config.get('ppo_epochs', 4)):
            # PPO更新 (log_stdのみ)
            ppo_loss, entropy_loss, kl_div = self._update_ppo_only(
                sequences, advantages_tensor, old_log_probs_tensor
            )
            
            total_policy_loss += ppo_loss
            total_entropy_loss += entropy_loss
            total_kl_div += kl_div
            
            # SmolVLA更新 (warmup後のみ)
            if self.should_update_smolvla(epoch):
                flow_matching_loss = self._update_smolvla_only(sequences)
                total_flow_matching_loss += flow_matching_loss
            
            # KLダイバージェンスが大きすぎる場合は早期停止
            # if kl_div > self.config.get('target_kl', 0.02):
            #     self.logger.info(f"Early stopping at PPO epoch {ppo_epoch} due to high KL divergence: {kl_div:.4f}")
            #     break
        
        num_epochs = ppo_epoch + 1
        avg_policy_loss = total_policy_loss / num_epochs
        avg_entropy_loss = total_entropy_loss / num_epochs
        avg_kl_div = total_kl_div / num_epochs
        avg_flow_matching_loss = total_flow_matching_loss / num_epochs if total_flow_matching_loss > 0 else 0.0
        
        return {
            'policy_loss': avg_policy_loss,
            'entropy_loss': avg_entropy_loss,
            'flow_matching_loss': avg_flow_matching_loss,
            'kl_divergence': avg_kl_div,
            'num_sequences': len(sequences),
            'avg_reward': np.mean([seq.reward for seq in sequences]),
            'avg_advantage': np.mean(advantages),
            'std_mean': torch.exp(self.policy.log_std).mean().item(),
            'smolvla_updated': self.should_update_smolvla(epoch)
        }
    
    def _update_ppo_only(self, sequences: List[ActionSequence], advantages_tensor: torch.Tensor, 
                        old_log_probs_tensor: torch.Tensor) -> Tuple[float, float, float]:
        """PPOの部分のみを更新（log_stdパラメータのみ）"""
        # 古い分布と新しい分布を作成
        old_distributions = []
        new_distributions = []
        new_log_probs = []
        entropies = []
        
        for seq in sequences:
            try:
                # 新しい分布を取得
                new_dist = self.policy.compute_action_distribution(seq.observation, seq.task)
                new_distributions.append(new_dist)
                
                # log probを計算
                log_prob = self.policy.compute_action_logprob(
                    seq.observation, seq.action, seq.task
                )
                new_log_probs.append(log_prob)
                
                # 古い分布を再構築（detachして勾配計算を無効化）
                with torch.no_grad():
                    old_mean = new_dist.mean.detach()
                    old_std = new_dist.stddev.detach()
                    old_dist = dist.Normal(old_mean, old_std)
                    old_distributions.append(old_dist)
                
                # エントロピー計算（PyTorchの分布から）
                entropy = new_dist.entropy().sum()
                entropies.append(entropy)
                
            except Exception as e:
                self.logger.warning(f"Failed to compute distributions: {e}")
                # デフォルト値を設定
                default_mean = torch.zeros(1, device=self.device, requires_grad=True)
                default_std = torch.ones(1, device=self.device, requires_grad=True)
                new_distributions.append(dist.Normal(default_mean, default_std))
                old_distributions.append(dist.Normal(default_mean.detach(), default_std.detach()))
                new_log_probs.append(torch.tensor(-10.0, device=self.device, requires_grad=True))
                entropies.append(torch.tensor(0.0, device=self.device, requires_grad=True))
        
        new_log_probs_tensor = torch.stack(new_log_probs)
        entropy_tensor = torch.stack(entropies)
        
        # log_probの差をクランプして数値的安定性を向上
        log_prob_diff = torch.clamp(new_log_probs_tensor - old_log_probs_tensor, min=-10.0, max=10.0)
        
        # 重要度サンプリング比
        ratio = torch.exp(log_prob_diff)
        
        # PyTorchのKLダイバージェンスを使用（必ず非負値）
        kl_divs = []
        for old_dist, new_dist in zip(old_distributions, new_distributions):
            try:
                kl_div_sample = dist.kl_divergence(old_dist, new_dist).sum()
                kl_divs.append(kl_div_sample)
            except Exception as e:
                self.logger.warning(f"Failed to compute KL divergence: {e}")
                kl_divs.append(torch.tensor(0.0, device=self.device))
        
        kl_div = torch.stack(kl_divs).mean()
        
        # 数値的安定性のためにクランプ（非負値を保証）
        kl_div = torch.clamp(kl_div, min=0.0, max=50.0)
        
        # PPO loss（数値的安定性を向上）
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(
            ratio, 
            1.0 - self.config['clip_epsilon'], 
            1.0 + self.config['clip_epsilon']
        ) * advantages_tensor
        
        policy_loss_raw = -torch.min(surr1, surr2).mean()
        entropy_loss_raw = -self.config.get('entropy_coef', 0.01) * entropy_tensor.mean()
        
        # 損失値をクランプして数値爆発を防止
        policy_loss = torch.clamp(policy_loss_raw, min=-1e6, max=1e6)
        entropy_loss = torch.clamp(entropy_loss_raw, min=-1e6, max=1e6)
        
        # PPOのみの更新（log_stdのみ）
        total_loss = policy_loss + entropy_loss
        
        self.ppo_optimizer.zero_grad()
        total_loss.backward()
        
        if self.config.get('max_grad_norm', 0) > 0:
            torch.nn.utils.clip_grad_norm_([self.policy.log_std], self.config['max_grad_norm'])
        
        self.ppo_optimizer.step()
        
        return policy_loss.item(), entropy_loss.item(), kl_div.item()
    
    def _update_smolvla_only(self, sequences: List[ActionSequence]) -> float:
        """SmolVLAの部分のみを更新（Flow Matching損失）"""
        flow_matching_loss = self.compute_flow_matching_loss(sequences)
        
        if flow_matching_loss.item() > 0:
            self.smolvla_optimizer.zero_grad()
            flow_matching_loss.backward()
            
            if self.config.get('max_grad_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.smolvla_policy.parameters(), 
                    self.config['max_grad_norm']
                )
            
            self.smolvla_optimizer.step()
        
        return flow_matching_loss.item()

    def update_policy(self, sequences: List[ActionSequence], advantages: np.ndarray) -> Dict:
        """レガシー互換性のため維持、内部でハイブリッド学習を呼び出し"""
        return self.update_policy_hybrid(sequences, advantages, self.epoch)
    
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
            log_dict = {
                'epoch': epoch,
                'value_loss': value_loss,
                'policy_loss': policy_update_info['policy_loss'],
                'avg_reward': np.mean([seq.reward for seq in sequences]),
                'num_sequences': len(sequences),
                'value_function_stable': self.is_value_function_stable(),
                'policy_updated': policy_update_info['policy_updated']
            }
            
            # 追加のメトリクスを含める
            if 'entropy_loss' in policy_update_info:
                log_dict.update({
                    'entropy_loss': policy_update_info['entropy_loss'],
                    'kl_divergence': policy_update_info['kl_divergence'],
                    'std_mean': policy_update_info['std_mean'],
                    'flow_matching_loss': policy_update_info.get('flow_matching_loss', 0.0),
                    'smolvla_updated': policy_update_info.get('smolvla_updated', False)
                })
            
            wandb.log(log_dict)
            
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

    def _upload_video(self, episode: int) -> None:
        """動画をWandBにアップロード"""
        try:
            if not self.video_frames:
                return
            
            # フレームを(T, H, W, C)形式に変換
            video_array = np.stack(self.video_frames, axis=0)  # (T, H, W, C)
            
            # 値の範囲を[0, 255]に正規化し、uint8に変換
            if video_array.max() <= 1.0:
                video_array = (video_array * 255).astype(np.uint8)
            else:
                video_array = video_array.astype(np.uint8)
            # THWC -> TCHW
            video_array = np.transpose(video_array, (0, 3, 1, 2))
            wandb.log({
                f"videos/video": wandb.Video(video_array, fps=30, format="mp4")
            })
            
            self.logger.info(f"Uploaded evaluation video for episode {episode}")
            
        except Exception as e:
            self.logger.warning(f"Failed to upload video: {e}")
    
    def save_checkpoint(self, epoch: int):
        checkpoint_dir = Path(self.config['checkpoint_dir']) / f"epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'policy_state_dict': self.policy.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'ppo_optimizer_state_dict': self.ppo_optimizer.state_dict(),
            'smolvla_optimizer_state_dict': self.smolvla_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'value_loss_history': self.value_loss_history,
            'config': self.config
        }
        
        policy_path = checkpoint_dir / "checkpoint.pth"
        torch.save(checkpoint, policy_path)
        
        smolvla_path = checkpoint_dir / "smolvla_policy"
        self.policy.smolvla_policy.save_pretrained(smolvla_path)
        wandb.save(str(policy_path))
