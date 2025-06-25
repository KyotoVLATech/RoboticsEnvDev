import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import torch.nn as nn
import numpy as np
import math
import logging
import torch.nn.functional as F

@dataclass
class DSRLExperience:
    """Single experience for DSRL training"""
    obs: Dict                     # 元の観測
    state_features: torch.Tensor  # VLMから抽出された状態特徴量
    latent_noise: torch.Tensor    # 潜在ノイズ（単一ステップ）
    action_chunk: torch.Tensor    # 生成された行動チャンク
    reward: float                 # 獲得報酬
    next_obs: Dict                # 次の状態の観測
    next_state_features: torch.Tensor  # 次の状態特徴量
    done: bool                    # エピソード終了フラグ
    task: str                     # タスク記述

class DSRLAgent(ABC):
    """
    DSRL強化学習エージェントの抽象基底クラス
    
    様々なRLアルゴリズム（NA、SAC、PPO等）に対応できるよう設計
    """
    
    def __init__(self, state_dim: int, noise_dim: int, config: Dict, device: str = "cuda"):
        self.state_dim = state_dim
        self.noise_dim = noise_dim
        self.config = config
        self.device = device
        
    @abstractmethod
    def select_noise(self, state_features: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """状態特徴量から潜在ノイズを選択"""
        pass
    
    @abstractmethod
    def update(self, experiences: List[DSRLExperience]) -> Dict[str, float]:
        """経験データからエージェントを更新"""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """チェックポイントを保存"""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """チェックポイントを読み込み"""
        pass


class LatentNoiseActor(nn.Module):
    """潜在ノイズを生成するアクターネットワーク"""
    
    def __init__(self, state_dim: int, noise_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.noise_dim = noise_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, noise_dim * 2)  # 平均と対数標準偏差
        )
        
        # 重みの初期化
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        状態特徴量から潜在ノイズの分布パラメータを予測
        
        Returns:
            mean: ノイズの平均
            log_std: ノイズの対数標準偏差
        """
        output = self.net(state_features)
        mean, log_std = output.chunk(2, dim=-1)
        
        # log_stdをクリップして数値安定性を確保
        log_std = torch.clamp(log_std, -5, 2)
        
        return mean, log_std
    
    def sample(self, state_features: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        潜在ノイズをサンプリング
        
        Returns:
            noise: サンプリングされた潜在ノイズ
            log_prob: ログ確率
        """
        mean, log_std = self.forward(state_features)
        
        if deterministic:
            noise = mean
            log_prob = torch.zeros_like(mean).sum(dim=-1)
        else:
            std = torch.exp(log_std)
            eps = torch.randn_like(mean)
            noise = mean + eps * std
            
            # ログ確率を計算（正規分布）
            log_prob = -0.5 * (((noise - mean) / std) ** 2 + 2 * log_std + math.log(2 * math.pi))
            log_prob = log_prob.sum(dim=-1)
        
        return noise, log_prob


class LatentNoiseCritic(nn.Module):
    """潜在ノイズ空間のQ関数"""
    
    def __init__(self, state_dim: int, noise_dim: int, hidden_dim: int = 1024):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 重みの初期化
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state_features: torch.Tensor, latent_noise: torch.Tensor) -> torch.Tensor:
        """
        状態特徴量と潜在ノイズからQ値を予測
        """
        x = torch.cat([state_features, latent_noise], dim=-1)
        return self.net(x).squeeze(-1)


class ActionCritic(nn.Module):
    """アクション空間のQ関数（DSRL-NAで使用）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 1024):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 重みの初期化
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state_features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        状態特徴量とアクションからQ値を予測
        
        Args:
            state_features: 状態特徴量 (batch_size, state_dim)
            actions: アクション (batch_size, chunk_size, action_dim) または (batch_size, action_dim)
        """
        if actions.dim() == 3:
            # Action chunkの場合、平均を取る
            actions = actions.mean(dim=1)
        
        x = torch.cat([state_features, actions], dim=-1)
        return self.net(x).squeeze(-1)

class DSRLNA(DSRLAgent):
    """
    DSRL-NA (Noise-Aliased) アルゴリズムの実装
    
    論文のAlgorithm 1に基づく実装：
    1. A-Critic: 元のアクション空間でのQ関数
    2. Latent-Noise Critic: 潜在ノイズ空間でのQ関数  
    3. Latent-Noise Actor: 最適な潜在ノイズを生成
    4. ノイズエイリアシングによる知識蒸留
    """
    
    def __init__(self, state_dim: int, noise_dim: int, action_dim: int, config: Dict, device: str = "cuda"):
        super().__init__(state_dim, noise_dim, config, device)
        
        self.action_dim = action_dim
        
        # SmolVLAWrapperへの参照（ノイズエイリアシングで使用）
        self.smolvla_wrapper = None
        
        # ネットワークの初期化
        hidden_dim = config.get('hidden_dim', 1024)
        
        # A-Critic: アクション空間のQ関数（2つでDouble Q-Learning）
        self.a_critic1 = ActionCritic(state_dim, action_dim, hidden_dim).to(device)
        self.a_critic2 = ActionCritic(state_dim, action_dim, hidden_dim).to(device)
        self.a_critic1_target = ActionCritic(state_dim, action_dim, hidden_dim).to(device)
        self.a_critic2_target = ActionCritic(state_dim, action_dim, hidden_dim).to(device)
        
        # ターゲットネットワークの初期化
        self.a_critic1_target.load_state_dict(self.a_critic1.state_dict())
        self.a_critic2_target.load_state_dict(self.a_critic2.state_dict())
        
        # Latent-Noise Critic: 潜在ノイズ空間のQ関数（2つでDouble Q-Learning）
        self.w_critic1 = LatentNoiseCritic(state_dim, noise_dim, hidden_dim).to(device)
        self.w_critic2 = LatentNoiseCritic(state_dim, noise_dim, hidden_dim).to(device)
        
        # Latent-Noise Actor: 潜在ノイズ生成ポリシー
        self.actor = LatentNoiseActor(state_dim, noise_dim, hidden_dim).to(device)
        
        # オプティマイザー
        lr = config.get('learning_rate', 3e-4)
        self.a_critic_optimizer = torch.optim.Adam(
            list(self.a_critic1.parameters()) + list(self.a_critic2.parameters()), lr=lr
        )
        self.w_critic_optimizer = torch.optim.Adam(
            list(self.w_critic1.parameters()) + list(self.w_critic2.parameters()), lr=lr
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # ハイパーパラメータ
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)  # ソフトターゲット更新率
        self.noise_std = config.get('noise_std', 0.1)  # 探索ノイズ
        self.target_update_freq = config.get('target_update_freq', 2)
        
        # 学習統計
        self.update_count = 0
        
        logging.info(f"DSRL-NA initialized: state_dim={state_dim}, noise_dim={noise_dim}, "
                    f"action_dim={action_dim}, hidden_dim={hidden_dim}")
    
    def set_smolvla_wrapper(self, smolvla_wrapper) -> None:
        """SmolVLAWrapperの参照を設定（ノイズエイリアシングで使用）"""
        self.smolvla_wrapper = smolvla_wrapper
    
    def select_noise(self, state_features: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """状態特徴量から潜在ノイズを選択"""
        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(0)
        
        with torch.no_grad():
            noise, _ = self.actor.sample(state_features, deterministic=deterministic)
        
        return noise.squeeze(0)
    
    def update(self, experiences: List[DSRLExperience]) -> Dict[str, float]:
        """経験データからDSRL-NAエージェントを更新"""
        if len(experiences) < 2:
            return {}
        
        # バッチデータの準備
        batch = self._prepare_batch(experiences)
        
        # 1. A-Criticの更新
        a_critic_loss = self._update_a_critic(batch)
        
        # 2. Latent-Noise Criticの更新（ノイズエイリアシング）
        w_critic_loss = self._update_w_critic(batch)
        
        # 3. Latent-Noise Actorの更新
        actor_loss = self._update_actor(batch)
        
        # 4. ターゲットネットワークの更新
        if self.update_count % self.target_update_freq == 0:
            self._update_target_networks()
        
        self.update_count += 1
        
        return {
            'a_critic_loss': a_critic_loss,
            'w_critic_loss': w_critic_loss,
            'actor_loss': actor_loss,
            'update_count': self.update_count
        }
    
    def _prepare_batch(self, experiences: List[DSRLExperience]) -> Dict[str, Any]:
        """経験リストをバッチテンソルに変換"""
        obs_list = [exp.obs for exp in experiences]
        state_features = torch.stack([exp.state_features for exp in experiences]).to(self.device)
        latent_noises = torch.stack([exp.latent_noise for exp in experiences]).to(self.device)
        action_chunks = torch.stack([exp.action_chunk for exp in experiences]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32).to(self.device)
        next_obs_list = [exp.next_obs for exp in experiences]
        next_state_features = torch.stack([exp.next_state_features for exp in experiences]).to(self.device)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool).to(self.device)
        tasks = [exp.task for exp in experiences]

        return {
            'obs': obs_list,
            'state_features': state_features,
            'latent_noises': latent_noises,
            'action_chunks': action_chunks,
            'rewards': rewards,
            'next_obs': next_obs_list,
            'next_state_features': next_state_features,
            'dones': dones,
            'tasks': tasks
        }
    
    def _update_a_critic(self, batch: Dict[str, Any]) -> float:
        """A-Critic（アクション空間Q関数）の更新"""
        state_features = batch['state_features']
        action_chunks = batch['action_chunks']
        rewards = batch['rewards']
        next_obs_list = batch['next_obs']
        next_state_features = batch['next_state_features']
        dones = batch['dones']
        tasks = batch['tasks']
        batch_size = state_features.shape[0]
        
        # 次の状態での行動を現在のアクターで生成（論文に基づく正しい実装）
        with torch.no_grad():
            next_noises, _ = self.actor.sample(next_state_features)
            
            # 次の状態でのアクションをSmolVLAで生成
            next_actions = []
            for i in range(batch_size):
                try:
                    # バッチから実際の次状態の観測を使用
                    next_obs = next_obs_list[i]
                    
                    # 次のノイズからSmolVLAでアクションを生成
                    next_action_chunk = self.smolvla_wrapper.generate_actions_from_noise(
                        next_state_features[i], next_noises[i], next_obs, tasks[i]
                    )
                    
                    # Action chunkの平均を取ってアクション次元に合わせる
                    next_action = next_action_chunk.mean(dim=0)
                    
                    # アクション次元を調整
                    if next_action.shape[0] > self.action_dim:
                        next_action = next_action[:self.action_dim]
                    elif next_action.shape[0] < self.action_dim:
                        next_action = F.pad(next_action, (0, self.action_dim - next_action.shape[0]))
                    
                    next_actions.append(next_action)
                    
                except Exception as e:
                    # SmolVLA生成に失敗した場合、現在のアクションで代替
                    logging.warning(f"Failed to generate next action in batch {i}: {e}")
                    current_action = action_chunks[i].mean(dim=0)
                    if current_action.shape[0] > self.action_dim:
                        current_action = current_action[:self.action_dim]
                    elif current_action.shape[0] < self.action_dim:
                        current_action = F.pad(current_action, (0, self.action_dim - current_action.shape[0]))
                    next_actions.append(current_action)
            
            # バッチ形式に変換
            next_actions = torch.stack(next_actions)
            
            # ターゲットQ値の計算
            target_q1 = self.a_critic1_target(next_state_features, next_actions)
            target_q2 = self.a_critic2_target(next_state_features, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + self.gamma * (1 - dones.float()) * target_q
        
        # 現在のQ値（action_chunksをaction次元に変換）
        current_actions = action_chunks.mean(dim=1) if action_chunks.dim() == 3 else action_chunks
        
        # アクション次元を調整
        if current_actions.shape[-1] > self.action_dim:
            current_actions = current_actions[:, :self.action_dim]
        elif current_actions.shape[-1] < self.action_dim:
            pad_size = self.action_dim - current_actions.shape[-1]
            current_actions = F.pad(current_actions, (0, pad_size))
        
        current_q1 = self.a_critic1(state_features, current_actions)
        current_q2 = self.a_critic2(state_features, current_actions)
        
        # A-CriticのLoss
        a_critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        
        # A-Criticの更新
        self.a_critic_optimizer.zero_grad()
        a_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.a_critic1.parameters()) + list(self.a_critic2.parameters()), 1.0
        )
        self.a_critic_optimizer.step()
        
        return a_critic_loss.item()

    def _update_w_critic(self, batch: Dict[str, Any]) -> float:
        """Latent-Noise Critic（潜在ノイズ空間Q関数）の更新 - ノイズエイリアシング"""
        obs_list = batch['obs']
        state_features = batch['state_features']
        tasks = batch['tasks']
        batch_size = state_features.shape[0]
        
        # ノイズエイリアシング：ランダムなノイズをサンプリング
        random_noises = torch.randn(batch_size, self.noise_dim, device=self.device)
        
        # 論文に基づく正しいノイズエイリアシング実装
        # ランダムノイズ → SmolVLA → 実際のアクション → A-Criticで評価
        with torch.no_grad():
            # SmolVLAWrapperを通して実際のアクションを生成
            # 注意: この処理は計算コストが高いため、バッチサイズを調整する場合あり
            aliased_actions = []
            for i in range(batch_size):
                try:
                    # バッチから実際の観測を使用
                    obs = obs_list[i]
                    
                    # ランダムノイズからSmolVLAでアクションを生成
                    action_chunk = self.smolvla_wrapper.generate_actions_from_noise(
                        state_features[i], random_noises[i], obs, tasks[i]
                    )
                    
                    # Action chunkの平均を取ってアクション次元に合わせる
                    aliased_action = action_chunk.mean(dim=0)
                    
                    # アクション次元を調整
                    if aliased_action.shape[0] > self.action_dim:
                        aliased_action = aliased_action[:self.action_dim]
                    elif aliased_action.shape[0] < self.action_dim:
                        aliased_action = F.pad(aliased_action, (0, self.action_dim - aliased_action.shape[0]))
                    
                    aliased_actions.append(aliased_action)
                    
                except Exception as e:
                    # SmolVLA生成に失敗した場合、ランダムアクションで代替
                    logging.warning(f"Failed to generate action from noise in batch {i}: {e}")
                    aliased_action = torch.randn(self.action_dim, device=self.device) * 0.1
                    aliased_actions.append(aliased_action)
            
            # バッチ形式に変換
            aliased_actions = torch.stack(aliased_actions)
            
            # A-Criticで価値評価
            target_q_values = torch.min(
                self.a_critic1(state_features, aliased_actions),
                self.a_critic2(state_features, aliased_actions)
            )
        
        # Latent-Noise CriticのQ値予測
        w_q1 = self.w_critic1(state_features, random_noises)
        w_q2 = self.w_critic2(state_features, random_noises)
        
        # 知識蒸留Loss（論文のAlgorithm 1に基づく）
        w_critic_loss = F.mse_loss(w_q1, target_q_values) + F.mse_loss(w_q2, target_q_values)
        
        # Latent-Noise Criticの更新
        self.w_critic_optimizer.zero_grad()
        w_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.w_critic1.parameters()) + list(self.w_critic2.parameters()), 1.0
        )
        self.w_critic_optimizer.step()
        
        return w_critic_loss.item()
    
    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> float:
        """Latent-Noise Actor（潜在ノイズ生成ポリシー）の更新"""
        state_features = batch['state_features']
        
        # アクターで潜在ノイズをサンプリング
        noises, log_probs = self.actor.sample(state_features)
        
        # Latent-Noise CriticでQ値を評価
        q1 = self.w_critic1(state_features, noises)
        q2 = self.w_critic2(state_features, noises)
        q_min = torch.min(q1, q2)
        
        # アクターLoss（Q値を最大化）
        actor_loss = -q_min.mean()
        
        # アクターの更新
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_target_networks(self):
        """ターゲットネットワークのソフト更新"""
        for param, target_param in zip(self.a_critic1.parameters(), self.a_critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.a_critic2.parameters(), self.a_critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_checkpoint(self, path: str) -> None:
        """チェックポイントを保存"""
        checkpoint = {
            'a_critic1_state_dict': self.a_critic1.state_dict(),
            'a_critic2_state_dict': self.a_critic2.state_dict(),
            'a_critic1_target_state_dict': self.a_critic1_target.state_dict(),
            'a_critic2_target_state_dict': self.a_critic2_target.state_dict(),
            'w_critic1_state_dict': self.w_critic1.state_dict(),
            'w_critic2_state_dict': self.w_critic2.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'a_critic_optimizer_state_dict': self.a_critic_optimizer.state_dict(),
            'w_critic_optimizer_state_dict': self.w_critic_optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'update_count': self.update_count,
            'config': self.config
        }
        torch.save(checkpoint, path)
        logging.info(f"DSRL-NA checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """チェックポイントを読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.a_critic1.load_state_dict(checkpoint['a_critic1_state_dict'])
        self.a_critic2.load_state_dict(checkpoint['a_critic2_state_dict'])
        self.a_critic1_target.load_state_dict(checkpoint['a_critic1_target_state_dict'])
        self.a_critic2_target.load_state_dict(checkpoint['a_critic2_target_state_dict'])
        self.w_critic1.load_state_dict(checkpoint['w_critic1_state_dict'])
        self.w_critic2.load_state_dict(checkpoint['w_critic2_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        
        self.a_critic_optimizer.load_state_dict(checkpoint['a_critic_optimizer_state_dict'])
        self.w_critic_optimizer.load_state_dict(checkpoint['w_critic_optimizer_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        
        self.update_count = checkpoint['update_count']
        
        logging.info(f"DSRL-NA checkpoint loaded from {path}")

class DSRLSAC(DSRLAgent):
    """
    DSRL-SAC アルゴリズムの実装
    
    標準的なSACを潜在ノイズ空間に適用したバージョン
    """
    
    def __init__(self, state_dim: int, noise_dim: int, config: Dict, device: str = "cuda"):
        super().__init__(state_dim, noise_dim, config, device)
        
        # ネットワークの初期化
        hidden_dim = config.get('hidden_dim', 1024)
        
        # Latent-Noise Critic: 潜在ノイズ空間のQ関数（2つでDouble Q-Learning）
        self.critic1 = LatentNoiseCritic(state_dim, noise_dim, hidden_dim).to(device)
        self.critic2 = LatentNoiseCritic(state_dim, noise_dim, hidden_dim).to(device)
        self.critic1_target = LatentNoiseCritic(state_dim, noise_dim, hidden_dim).to(device)
        self.critic2_target = LatentNoiseCritic(state_dim, noise_dim, hidden_dim).to(device)
        
        # ターゲットネットワークの初期化
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Latent-Noise Actor: 潜在ノイズ生成ポリシー
        self.actor = LatentNoiseActor(state_dim, noise_dim, hidden_dim).to(device)
        
        # エントロピー係数（自動調整）
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.target_entropy = -noise_dim  # SAC標準の目標エントロピー
        
        # オプティマイザー
        lr = config.get('learning_rate', 3e-4)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        # ハイパーパラメータ
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)  # ソフトターゲット更新率
        self.target_update_freq = config.get('target_update_freq', 2)
        
        # 学習統計
        self.update_count = 0
        
        logging.info(f"DSRL-SAC initialized: state_dim={state_dim}, noise_dim={noise_dim}, "
                    f"hidden_dim={hidden_dim}")
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_noise(self, state_features: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """状態特徴量から潜在ノイズを選択"""
        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(0)
        
        with torch.no_grad():
            noise, _ = self.actor.sample(state_features, deterministic=deterministic)
        
        return noise.squeeze(0)
    
    def update(self, experiences: List[DSRLExperience]) -> Dict[str, float]:
        """経験データからDSRL-SACエージェントを更新"""
        if len(experiences) < 2:
            return {}
        
        # バッチデータの準備
        batch = self._prepare_batch(experiences)
        
        # 1. Criticの更新
        critic_loss = self._update_critic(batch)
        
        # 2. Actorの更新
        actor_loss, entropy = self._update_actor(batch)
        
        # 3. エントロピー係数の更新
        alpha_loss = self._update_alpha(entropy)
        
        # 4. ターゲットネットワークの更新
        if self.update_count % self.target_update_freq == 0:
            self._update_target_networks()
        
        self.update_count += 1
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha.item(),
            'entropy': entropy,
            'update_count': self.update_count
        }
    
    def _prepare_batch(self, experiences: List[DSRLExperience]) -> Dict[str, Any]:
        """経験リストをバッチテンソルに変換"""
        obs_list = [exp.obs for exp in experiences]
        state_features = torch.stack([exp.state_features for exp in experiences]).to(self.device)
        latent_noises = torch.stack([exp.latent_noise for exp in experiences]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32).to(self.device)
        next_obs_list = [exp.next_obs for exp in experiences]
        next_state_features = torch.stack([exp.next_state_features for exp in experiences]).to(self.device)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool).to(self.device)
        
        return {
            'obs': obs_list,
            'state_features': state_features,
            'latent_noises': latent_noises,
            'rewards': rewards,
            'next_obs': next_obs_list,
            'next_state_features': next_state_features,
            'dones': dones
        }
    
    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> float:
        """Critic（潜在ノイズ空間Q関数）の更新"""
        state_features = batch['state_features']
        latent_noises = batch['latent_noises']
        rewards = batch['rewards']
        next_state_features = batch['next_state_features']
        dones = batch['dones']
        
        # 次の状態での行動とQ値を計算
        with torch.no_grad():
            next_noises, next_log_probs = self.actor.sample(next_state_features)
            
            # ターゲットQ値の計算
            target_q1 = self.critic1_target(next_state_features, next_noises)
            target_q2 = self.critic2_target(next_state_features, next_noises)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target = rewards + self.gamma * (1 - dones.float()) * target_q
        
        # 現在のQ値
        current_q1 = self.critic1(state_features, latent_noises)
        current_q2 = self.critic2(state_features, latent_noises)
        
        # CriticのLoss
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        
        # Criticの更新
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 1.0
        )
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """Actor（潜在ノイズ生成ポリシー）の更新"""
        state_features = batch['state_features']
        
        # アクターで潜在ノイズをサンプリング
        noises, log_probs = self.actor.sample(state_features)
        
        # CriticでQ値を評価
        q1 = self.critic1(state_features, noises)
        q2 = self.critic2(state_features, noises)
        q_min = torch.min(q1, q2)
        
        # アクターLoss（SAC標準）
        actor_loss = (self.alpha * log_probs - q_min).mean()
        
        # アクターの更新
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return actor_loss.item(), log_probs.mean().item()
    
    def _update_alpha(self, entropy: float) -> float:
        """エントロピー係数の自動調整"""
        alpha_loss = -(self.log_alpha * (entropy + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return alpha_loss.item()
    
    def _update_target_networks(self):
        """ターゲットネットワークのソフト更新"""
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_checkpoint(self, path: str) -> None:
        """チェックポイントを保存"""
        checkpoint = {
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'log_alpha': self.log_alpha.item(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'update_count': self.update_count,
            'config': self.config
        }
        torch.save(checkpoint, path)
        logging.info(f"DSRL-SAC checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """チェックポイントを読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        
        self.log_alpha.data.fill_(checkpoint['log_alpha'])
        
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.update_count = checkpoint['update_count']
        
        logging.info(f"DSRL-SAC checkpoint loaded from {path}")