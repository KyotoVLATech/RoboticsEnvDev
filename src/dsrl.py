"""
DSRL (Diffusion Steering via Reinforcement Learning) Framework for SmolVLA

This module implements the DSRL framework as described in the paper
"Steering Your Diffusion Policy with Latent Space Reinforcement Learning"
adapted for LeRobot's SmolVLA model.

The framework treats the pre-trained SmolVLA model as a black box and learns
to control its behavior by manipulating the latent noise input to the Flow Matching
action generation process.
"""

import os
import sys
import time
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.genesis_env import GenesisEnv
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from src.make_sim_dataset import task_description


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


class SmolVLAWrapper:
    """
    SmolVLAモデルをDSRLで制御するためのラッパークラス
    
    DSRLの核心概念：
    1. SmolVLAモデルを凍結してブラックボックスとして扱う
    2. Flow Matchingの潜在ノイズをRLで制御する
    3. Action Chunkingに対応（単一ノイズをchunk_sizeにコピー）
    """
    
    def __init__(self, smolvla_policy: SmolVLAPolicy, device: str = "cuda"):
        self.smolvla_policy = smolvla_policy.to(device)
        self.device = device
        
        # SmolVLAモデルを凍結（ブラックボックス化）
        for param in self.smolvla_policy.parameters():
            param.requires_grad = False
        self.smolvla_policy.eval()
        
        # 設定値を取得
        self.chunk_size = getattr(self.smolvla_policy.config, 'chunk_size', 50)
        self.max_action_dim = getattr(self.smolvla_policy.config, 'max_action_dim', 32)
        self.noise_dim = self.max_action_dim  # 潜在ノイズの次元
        
        # VLMの隠れ状態次元を取得
        self.vlm_hidden_size = self.smolvla_policy.model.vlm_with_expert.config.text_config.hidden_size
        
        # 論文に基づく状態特徴量の次元を計算
        # 1. 自己受容状態の次元
        self.proprioceptive_dim = getattr(self.smolvla_policy.config, 'max_state_dim', 32)
        
        # 2. VLM最終トークン特徴量の次元（論文では2048次元）
        self.vlm_final_token_dim = self.vlm_hidden_size
        
        # 3. 視覚特徴量の次元（簡易実装：画像チャンネル数 * カメラ数）
        # 論文では浅い畳み込みエンコーダを使用するが、ここでは簡易的に設定
        self.visual_features_dim = 3 * 2  # RGB * 2カメラ（front, side）
        
        # 総合的な状態特徴量の次元
        self.total_state_dim = (
            self.proprioceptive_dim + 
            self.vlm_final_token_dim + 
            self.visual_features_dim
        )
        
        logging.info(f"SmolVLAWrapper initialized:")
        logging.info(f"  chunk_size={self.chunk_size}")
        logging.info(f"  noise_dim={self.noise_dim}")
        logging.info(f"  vlm_hidden_size={self.vlm_hidden_size}")
        logging.info(f"  proprioceptive_dim={self.proprioceptive_dim}")
        logging.info(f"  vlm_final_token_dim={self.vlm_final_token_dim}")
        logging.info(f"  visual_features_dim={self.visual_features_dim}")
        logging.info(f"  total_state_dim={self.total_state_dim}")
    
    def extract_state_features(self, obs: Dict, task: str) -> torch.Tensor:
        """
        論文に基づく状態特徴量の抽出
        
        論文の記述：
        "We input the proprioceptive state, the final token's last hidden feature 
        from π0's VLM backbone (a 2,048-dimensional vector), and visual features 
        into the noise policy."
        
        SmolVLA.sample_actionsの実装を参考に、実際にVLMの順伝播を行って
        最終トークンの隠れ特徴量を取得する。
        
        Args:
            obs: 環境からの観測
            task: タスク記述
        
        Returns:
            torch.Tensor: 論文に基づく状態特徴量
        """
        with torch.no_grad():
            batch = self._prepare_batch(obs, task)
            
            # 1. 自己受容状態（proprioceptive state）の取得
            proprioceptive_state = self.smolvla_policy.prepare_state(batch)  # (1, state_dim)
            
            # 2. VLMバックボーンの最終トークンの隠れ特徴量を正しく取得
            images, img_masks = self.smolvla_policy.prepare_images(batch)
            lang_tokens, lang_masks = self.smolvla_policy.prepare_language(batch)
            
            # SmolVLA.sample_actionsと同様の処理でVLMの隠れ特徴量を取得
            prefix_embs, prefix_pad_masks, prefix_att_masks = \
                self.smolvla_policy.model.embed_prefix(
                    images, img_masks, lang_tokens, lang_masks, state=proprioceptive_state
                )
            
            # VLMの順伝播を行って隠れ特徴量を取得
            # make_att_2d_masksをインポートして使用
            from lerobot.common.policies.smolvla.modeling_smolvla import make_att_2d_masks
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            
            # VLMの順伝播を実行して隠れ特徴量を取得
            vlm_outputs, _ = self.smolvla_policy.model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
                fill_kv_cache=True,
            )
            
            # VLM出力から最終トークンの隠れ特徴量を抽出
            prefix_hidden_states = vlm_outputs[0]  # (batch_size, seq_len, hidden_dim)
            
            # 論文に基づく：最終トークンの隠れ特徴量を取得（2,048次元）
            valid_mask = prefix_pad_masks.bool()
            # 各バッチについて最後の有効なトークンのインデックスを見つける
            last_valid_indices = valid_mask.sum(dim=1) - 1  # (batch_size,)
            batch_indices = torch.arange(prefix_hidden_states.shape[0], device=prefix_hidden_states.device)
            final_token_features = prefix_hidden_states[batch_indices, last_valid_indices]  # (batch_size, hidden_dim)
            
            # 3. 視覚特徴量の取得（論文に基づく簡易実装）
            # 論文では複数カメラからの浅い畳み込みエンコーダを使用
            # ここではSmolVLAの画像エンコーダ特徴量を使用
            visual_features_list = []
            for key in ['observation.images.front', 'observation.images.side']:
                if key in batch:
                    # SmolVLAの画像エンコーダから特徴量を抽出
                    img_tensor = batch[key]  # (1, C, H, W)
                    # 簡易的にGlobal Average Poolingで特徴量を取得
                    img_features = F.adaptive_avg_pool2d(img_tensor, (1, 1)).flatten(1)  # (1, C)
                    visual_features_list.append(img_features)
            
            if visual_features_list:
                visual_features = torch.cat(visual_features_list, dim=-1)  # 複数画像を結合
            else:
                # 画像がない場合はゼロベクトル
                visual_features = torch.zeros(1, self.visual_features_dim, device=self.device)
            
            # 視覚特徴量の次元を調整
            if visual_features.shape[-1] != self.visual_features_dim:
                if visual_features.shape[-1] > self.visual_features_dim:
                    visual_features = visual_features[:, :self.visual_features_dim]
                else:
                    pad_size = self.visual_features_dim - visual_features.shape[-1]
                    visual_features = F.pad(visual_features, (0, pad_size))
            
            # 4. 全ての特徴量を結合
            # - proprioceptive_state: (1, state_dim)
            # - final_token_features: (1, hidden_dim) ≈ 2048次元
            # - visual_features: (1, visual_dim)
            
            state_features = torch.cat([
                proprioceptive_state.flatten(),      # 自己受容状態
                final_token_features.flatten(),      # VLM最終トークン特徴量（実際のVLM出力）
                visual_features.flatten()            # 視覚特徴量
            ], dim=0)
            
            return state_features
    
    def generate_actions_from_noise(self, state_features: torch.Tensor, 
                                  latent_noise: torch.Tensor, 
                                  obs: Dict, task: str) -> torch.Tensor:
        """
        潜在ノイズから行動チャンクを生成
        
        Args:
            state_features: VLMから抽出された状態特徴量
            latent_noise: 潜在ノイズ（単一ステップ）
            obs: 元の観測（画像等の再構築用）
            task: タスク記述
        
        Returns:
            torch.Tensor: 生成された行動チャンク
        """
        with torch.no_grad():
            batch = self._prepare_batch(obs, task)
            
            # 画像とlanguageの準備
            images, img_masks = self.smolvla_policy.prepare_images(batch)
            state = self.smolvla_policy.prepare_state(batch)
            lang_tokens, lang_masks = self.smolvla_policy.prepare_language(batch)
            
            # 論文の戦略：単一ステップのノイズをchunk_sizeにコピー
            if latent_noise.dim() == 1:
                latent_noise = latent_noise.unsqueeze(0)  # バッチ次元を追加
            
            # (batch_size, 1, noise_dim) -> (batch_size, chunk_size, noise_dim)
            noise_chunk = latent_noise.unsqueeze(1).repeat(1, self.chunk_size, 1)
            
            # SmolVLAのsample_actionsの内部処理を模倣
            actions = self.smolvla_policy.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise_chunk
            )
            
            # 元のaction次元に切り取り
            original_action_dim = self.smolvla_policy.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]
            
            # 正規化を解除
            actions = self.smolvla_policy.unnormalize_outputs({"action": actions})["action"]
            
            # Aloha環境の場合の変換
            if getattr(self.smolvla_policy.config, 'adapt_to_pi_aloha', False):
                actions = self.smolvla_policy._pi_aloha_encode_actions(actions)
            
            return actions.squeeze(0)  # バッチ次元を除去
    
    def _prepare_batch(self, obs: Dict, task: str) -> Dict[str, torch.Tensor]:
        """観測をSmolVLA用のバッチ形式に変換"""
        batch = {}
        
        # 状態情報 - agent_posを観測状態として使用
        if 'agent_pos' in obs:
            agent_pos = obs['agent_pos']
            if isinstance(agent_pos, np.ndarray):
                agent_pos = torch.from_numpy(agent_pos.copy()).float()
            else:
                agent_pos = agent_pos.float()
            
            if agent_pos.dim() == 1:
                agent_pos = agent_pos.unsqueeze(0)
            batch['observation.state'] = agent_pos.to(self.device)
        
        # 画像情報 - GenesisEnvの観測形式に合わせて処理
        for key in ['observation.images.front', 'observation.images.side']:
            if key in obs:
                img = obs[key]
                if isinstance(img, np.ndarray):
                    # NumPy配列の場合
                    img = torch.from_numpy(img.copy()).float()
                else:
                    # すでにTensorの場合
                    img = img.float()
                
                # 正規化 (0-255 -> 0-1)
                if img.max() > 1.0:
                    img = img / 255.0
                
                # 次元の順序を調整: (H, W, C) -> (C, H, W)
                if img.ndim == 3 and img.shape[2] in [1, 3, 4]:
                    img = img.permute(2, 0, 1)
                elif img.ndim == 2:
                    img = img.unsqueeze(0)
                
                # バッチ次元を追加して保存
                batch[key] = img.to(self.device).unsqueeze(0)
        
        # タスク記述
        batch['task'] = task_description.get(task, task)
        
        return batch


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


class ReplayBuffer:
    """DSRL用のリプレイバッファ"""
    
    def __init__(self, capacity: int, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: DSRLExperience):
        """経験を追加"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[DSRLExperience]:
        """バッチサイズ分の経験をサンプリング"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float = 1.0, max_period: float = 1000.0, device="cpu"
) -> torch.Tensor:
    """時間エンベッディング用の正弦波位置エンコーディング"""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # スケーリング係数を計算
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    """ベクトルを指定次元までパディング"""
    if vector.shape[-1] == new_dim:
        return vector
    
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


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
        self.smolvla_wrapper = None  # DSRLTrainerで設定される
        
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
    
    def set_smolvla_wrapper(self, smolvla_wrapper: 'SmolVLAWrapper') -> None:
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

class DSRLTrainer:
    """
    DSRLフレームワークの学習ループを管理するトレーナークラス
    """
    
    def __init__(self, env: GenesisEnv, smolvla_wrapper: SmolVLAWrapper, 
                 dsrl_agent: DSRLAgent, config: Dict, device: str = "cuda"):
        self.env = env
        self.smolvla_wrapper = smolvla_wrapper
        self.dsrl_agent = dsrl_agent
        self.config = config
        self.device = device
        
        # リプレイバッファ
        self.replay_buffer = ReplayBuffer(
            capacity=config.get('replay_buffer_size', 100000),
            device=device
        )
        
        # 学習統計
        self.episode_count = 0
        self.step_count = 0
        self.total_rewards = []
        self.success_rates = []
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        
        # Wandb初期化
        if config.get('use_wandb', True):
            wandb.init(
                project=config.get('wandb_project', 'dsrl-smolvla'),
                name=config.get('wandb_run_name', f"dsrl_{config['algorithm']}_{config['task']}"),
                config=config,
                sync_tensorboard=False
            )
    
    def train(self) -> None:
        """メインの学習ループ"""
        self.logger.info(f"Starting DSRL training for {self.config['total_episodes']} episodes")
        
        for episode in range(self.config['total_episodes']):
            episode_reward, episode_length, success = self._run_episode()
            
            self.total_rewards.append(episode_reward)
            self.success_rates.append(float(success))
            self.episode_count += 1
            
            # ログ記録
            if episode % self.config.get('log_freq', 10) == 0:
                avg_reward = np.mean(self.total_rewards[-10:])
                avg_success = np.mean(self.success_rates[-10:])
                
                self.logger.info(
                    f"Episode {episode}: reward={episode_reward:.3f}, "
                    f"length={episode_length}, success={success}, "
                    f"avg_reward={avg_reward:.3f}, avg_success={avg_success:.3f}"
                )
                
                if self.config.get('use_wandb', True):
                    wandb.log({
                        'episode': episode,
                        'episode_reward': episode_reward,
                        'episode_length': episode_length,
                        'episode_success': float(success),
                        'avg_reward_10': avg_reward,
                        'avg_success_10': avg_success,
                        'replay_buffer_size': len(self.replay_buffer)
                    })
            
            # 学習の実行
            if len(self.replay_buffer) >= self.config.get('min_replay_size', 1000):
                for _ in range(self.config.get('updates_per_episode', 1)):
                    self._update_agent()
            
            # チェックポイント保存
            if episode % self.config.get('save_freq', 100) == 0:
                self._save_checkpoint(episode)
        
        # 最終チェックポイント保存
        self._save_checkpoint(self.config['total_episodes'])
        
        if self.config.get('use_wandb', True):
            wandb.finish()
    
    def _run_episode(self) -> Tuple[float, int, bool]:
        """1エピソードを実行"""
        obs, info = self.env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        # 前の状態を初期化
        prev_obs = None
        prev_state_features = None
        prev_noise = None
        prev_task = None
        
        while not done and episode_length < self.config.get('max_episode_length', 500):
            # 現在の状態特徴量を抽出
            current_state_features = self.smolvla_wrapper.extract_state_features(obs, self.config['task'])
            
            # DSRLエージェントで潜在ノイズを選択
            deterministic = self.config.get('deterministic_eval', False) and \
                          self.episode_count % self.config.get('eval_freq', 50) == 0
            latent_noise = self.dsrl_agent.select_noise(current_state_features, deterministic=deterministic)
            
            # SmolVLAで行動チャンクを生成
            action_chunk = self.smolvla_wrapper.generate_actions_from_noise(
                current_state_features, latent_noise, obs, self.config['task']
            )
            
            # Action chunkを逐次実行
            chunk_reward = 0.0
            chunk_length = 0
            actions_executed = []
            
            for action_idx in range(min(self.config.get('chunk_size', 50), len(action_chunk))):
                if done:
                    break
                
                action = action_chunk[action_idx]
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                chunk_reward += reward
                chunk_length += 1
                episode_length += 1
                self.step_count += 1
                
                actions_executed.append(action)
                
                done = terminated or truncated
                if done:
                    break
                
                obs = next_obs
            
            episode_reward += chunk_reward
            
            # 経験をリプレイバッファに追加（前のステップがある場合）
            if prev_obs is not None:
                executed_actions = np.array(actions_executed)
                if len(executed_actions) > 0:
                    # アクションチャンクの形に合わせる
                    if executed_actions.ndim == 1:
                        executed_actions = executed_actions.reshape(1, -1)
                    
                    experience = DSRLExperience(
                        obs=prev_obs,
                        state_features=prev_state_features,
                        latent_noise=prev_noise,
                        action_chunk=torch.from_numpy(executed_actions).float().to(self.device),
                        reward=chunk_reward / len(actions_executed) if len(actions_executed) > 0 else 0.0,
                        next_obs=obs.copy() if hasattr(obs, 'copy') else obs,
                        next_state_features=current_state_features,
                        done=done,
                        task=prev_task
                    )
                    self.replay_buffer.add(experience)
            
            # 次のステップのために更新
            prev_obs = obs.copy() if hasattr(obs, 'copy') else obs
            prev_state_features = current_state_features
            prev_noise = latent_noise
            prev_task = self.config['task']
            
            if done:
                break
        
        success = info.get('is_success', False)
        return episode_reward, episode_length, success
    
    def _update_agent(self) -> None:
        """DSRLエージェントを更新"""
        if len(self.replay_buffer) < self.config.get('batch_size', 64):
            return
        
        # バッチサンプリング
        experiences = self.replay_buffer.sample(self.config.get('batch_size', 64))
        
        # エージェント更新
        update_info = self.dsrl_agent.update(experiences)
        
        # Wandbログ
        if self.config.get('use_wandb', True) and update_info:
            wandb_log = {f"agent/{k}": v for k, v in update_info.items()}
            wandb.log(wandb_log)
    
    def _save_checkpoint(self, episode: int) -> None:
        """チェックポイントを保存"""
        checkpoint_dir = Path(self.config['checkpoint_dir']) / f"episode_{episode}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # DSRLエージェントのチェックポイント
        agent_path = checkpoint_dir / "dsrl_agent.pth"
        self.dsrl_agent.save_checkpoint(str(agent_path))
        
        # 学習統計の保存
        stats = {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'total_rewards': self.total_rewards,
            'success_rates': self.success_rates,
            'config': self.config
        }
        stats_path = checkpoint_dir / "training_stats.pth"
        torch.save(stats, stats_path)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")

def create_dsrl_agent(algorithm: str, state_dim: int, noise_dim: int, action_dim: int, 
                     config: Dict, device: str = "cuda") -> DSRLAgent:
    """
    指定されたアルゴリズムでDSRLエージェントを作成
    
    Args:
        algorithm: "NA" または "SAC"
        state_dim: 状態特徴量の次元
        noise_dim: 潜在ノイズの次元
        action_dim: アクションの次元（DSRL-NAでのみ使用）
        config: 設定辞書
        device: デバイス
    
    Returns:
        DSRLAgent: 指定されたアルゴリズムのエージェント
    """
    if algorithm.upper() == "NA":
        return DSRLNA(state_dim, noise_dim, action_dim, config, device)
    elif algorithm.upper() == "SAC":
        return DSRLSAC(state_dim, noise_dim, config, device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: 'NA', 'SAC'")

def load_smolvla_model(model_path: Optional[str] = None, config_overrides: Optional[Dict] = None) -> SmolVLAPolicy:
    """
    SmolVLAモデルを読み込み
    
    Args:
        model_path: 事前学習済みモデルのパス（Noneの場合はデフォルト設定）
        config_overrides: 設定のオーバーライド
    
    Returns:
        SmolVLAPolicy: SmolVLAポリシー
    """
    if model_path and Path(model_path).exists():
        # 事前学習済みモデルを読み込み
        policy = SmolVLAPolicy.from_pretrained(model_path)
        logging.info(f"Loaded pre-trained SmolVLA model from {model_path}")
    else:
        # デフォルト設定でモデルを作成
        config = SmolVLAConfig()
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)
        policy = SmolVLAPolicy(config)
        logging.info("Created SmolVLA model with default configuration")
    
    return policy

def main():
    """メイン実行関数"""
    
    # 設定
    config = {
        # 環境設定
        'task': 'simple_pick',
        'observation_height': 512,
        'observation_width': 512,
        'show_viewer': False,
        
        # DSRL設定
        'algorithm': 'NA',  # 'NA' または 'SAC'
        'total_episodes': 2000,
        'max_episode_length': 500,
        'chunk_size': 50,
        
        # ネットワーク設定
        'hidden_dim': 1024,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'target_update_freq': 2,
        
        # 学習設定
        'batch_size': 64,
        'replay_buffer_size': 100000,
        'min_replay_size': 1000,
        'updates_per_episode': 1,
        
        # ログ・保存設定
        'use_wandb': True,
        'wandb_project': 'dsrl-smolvla',
        'wandb_run_name': None,  # Noneの場合は自動生成
        'log_freq': 10,
        'save_freq': 100,
        'eval_freq': 50,
        'deterministic_eval': False,
        'checkpoint_dir': 'outputs/train/dsrl_checkpoints',
        
        # SmolVLA設定
        'pretrained_model_path': "lerobot/smolvla_base",  # 事前学習済みモデルのパス
        'smolvla_config_overrides': {
            'chunk_size': 50,
            'max_action_dim': 7,  # Genesis環境のaction次元に合わせて調整
        }
    }
    
    # コマンドライン引数の処理（簡易版）
    import sys
    for i, arg in enumerate(sys.argv):
        if arg == '--algorithm' and i + 1 < len(sys.argv):
            config['algorithm'] = sys.argv[i + 1]
        elif arg == '--task' and i + 1 < len(sys.argv):
            config['task'] = sys.argv[i + 1]
        elif arg == '--episodes' and i + 1 < len(sys.argv):
            config['total_episodes'] = int(sys.argv[i + 1])
        elif arg == '--pretrained' and i + 1 < len(sys.argv):
            config['pretrained_model_path'] = sys.argv[i + 1]
        elif arg == '--no-wandb':
            config['use_wandb'] = False
    
    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # 出力ディレクトリの作成
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    # 環境の作成
    env = GenesisEnv(
        task=config['task'],
        observation_height=config['observation_height'],
        observation_width=config['observation_width'],
        show_viewer=config['show_viewer']
    )
    logging.info(f"Created environment for task: {config['task']}")
    
    # SmolVLAモデルの読み込み
    smolvla_policy = load_smolvla_model(
        config['pretrained_model_path'],
        config['smolvla_config_overrides']
    )
    
    # SmolVLAラッパーの作成
    smolvla_wrapper = SmolVLAWrapper(smolvla_policy, device)
    
    # 次元設定（論文に基づく修正）
    state_dim = smolvla_wrapper.total_state_dim  # 論文に基づく総合状態特徴量次元
    noise_dim = smolvla_wrapper.noise_dim
    action_dim = config['smolvla_config_overrides']['max_action_dim']
    
    logging.info(f"Dimensions - state: {state_dim}, noise: {noise_dim}, action: {action_dim}")
    logging.info(f"State composition:")
    logging.info(f"  - proprioceptive: {smolvla_wrapper.proprioceptive_dim}")
    logging.info(f"  - vlm_final_token: {smolvla_wrapper.vlm_final_token_dim}")
    logging.info(f"  - visual_features: {smolvla_wrapper.visual_features_dim}")
    
    # DSRLエージェントの作成
    dsrl_agent = create_dsrl_agent(
        config['algorithm'], state_dim, noise_dim, action_dim, config, device
    )
    
    # DSRL-NAエージェントの場合、SmolVLAWrapperへの参照を設定
    if hasattr(dsrl_agent, 'set_smolvla_wrapper'):
        dsrl_agent.set_smolvla_wrapper(smolvla_wrapper)
    
    # Wandb実行名の設定
    if config['wandb_run_name'] is None:
        config['wandb_run_name'] = f"dsrl_{config['algorithm'].lower()}_{config['task']}"
    
    # トレーナーの作成と学習実行
    trainer = DSRLTrainer(env, smolvla_wrapper, dsrl_agent, config, device)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        trainer._save_checkpoint(trainer.episode_count)
    finally:
        env.close()
        logging.info("Training completed")

def evaluate_dsrl_model(checkpoint_path: str, config: Dict, num_episodes: int = 10):
    """
    学習済みDSRLモデルの評価
    
    Args:
        checkpoint_path: DSRLエージェントのチェックポイントパス
        config: 設定辞書
        num_episodes: 評価エピソード数
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 環境の作成
    env = GenesisEnv(
        task=config['task'],
        observation_height=config['observation_height'],
        observation_width=config['observation_width'],
        show_viewer=True  # 評価時は表示
    )
    
    # SmolVLAモデルの読み込み
    smolvla_policy = load_smolvla_model(
        config['pretrained_model_path'],
        config['smolvla_config_overrides']
    )
    smolvla_wrapper = SmolVLAWrapper(smolvla_policy, device)
    
    # DSRLエージェントの作成と読み込み（論文に基づく修正）
    state_dim = smolvla_wrapper.total_state_dim  # 論文に基づく総合状態特徴量次元
    noise_dim = smolvla_wrapper.noise_dim
    action_dim = config['smolvla_config_overrides']['max_action_dim']
    
    dsrl_agent = create_dsrl_agent(
        config['algorithm'], state_dim, noise_dim, action_dim, config, device
    )
    dsrl_agent.load_checkpoint(checkpoint_path)
    
    # 評価実行
    total_rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done and episode_length < config.get('max_episode_length', 500):
            # 状態特徴量を抽出
            state_features = smolvla_wrapper.extract_state_features(obs, config['task'])
            
            # 決定論的にノイズを選択
            latent_noise = dsrl_agent.select_noise(state_features, deterministic=True)
            
            # 行動チャンクを生成
            action_chunk = smolvla_wrapper.generate_actions_from_noise(
                state_features, latent_noise, obs, config['task']
            )
            
            # Action chunkを実行
            for action_idx in range(min(config.get('chunk_size', 50), len(action_chunk))):
                if done:
                    break
                
                action = action_chunk[action_idx]
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                done = terminated or truncated
                if done:
                    break
        
        total_rewards.append(episode_reward)
        if info.get('is_success', False):
            success_count += 1
        
        logging.info(f"Episode {episode + 1}: reward={episode_reward:.3f}, "
                    f"length={episode_length}, success={info.get('is_success', False)}")
    
    # 結果の表示
    avg_reward = np.mean(total_rewards)
    success_rate = success_count / num_episodes
    
    logging.info(f"\nEvaluation Results ({num_episodes} episodes):")
    logging.info(f"Average Reward: {avg_reward:.3f}")
    logging.info(f"Success Rate: {success_rate:.3f} ({success_count}/{num_episodes})")
    
    env.close()
    return avg_reward, success_rate

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    main()
