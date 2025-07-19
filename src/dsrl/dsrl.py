"""
DSRL (Diffusion Steering via Reinforcement Learning) Framework for SmolVLA

This module implements the DSRL framework as described in the paper
"Steering Your Diffusion Policy with Latent Space Reinforcement Learning"
adapted for LeRobot's SmolVLA model.

The framework treats the pre-trained SmolVLA model as a black box and learns
to control its behavior by manipulating the latent noise input to the Flow Matching
action generation process.

NOTE: This module now only contains the SmolVLAWrapper class.
For training, use train_dsrl_tianshou.py which implements the full training loop
using tianshou's SAC algorithm and the NoiseActionEnv wrapper.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

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
        
        # 2. テキスト特徴量の次元（ImageValueNetworkと同様）
        self.text_features_dim = 960
        
        # 3. 視覚特徴量の次元（ImageValueNetworkと同様）
        self.visual_features_dim = 960 * 2
        
        # 総合的な状態特徴量の次元
        self.total_state_dim = (
            self.proprioceptive_dim + 
            self.text_features_dim + 
            self.visual_features_dim
        )

    def extract_state_features(self, obs: Dict, task_desc: str) -> torch.Tensor:
        """
        ImageValueNetworkを参考にした効率的な状態特徴量の抽出
        Args:
            obs: 環境からの観測
            task_desc: タスク記述
        Returns:
            torch.Tensor: 統合された状態特徴量
        """
        with torch.no_grad():
            batch = self._prepare_batch(obs, task_desc)
            # 1. 自己受容状態（proprioceptive state）の取得
            proprioceptive_state = self.smolvla_policy.prepare_state(batch)  # (1, state_dim)
            # 2. テキスト特徴量の抽出（ImageValueNetworkと同様）
            text_features = self._extract_text_features(batch)
            # 3. 視覚特徴量の抽出（ImageValueNetworkと同様）
            visual_features = self._extract_vision_features(batch)
            # 4. 全ての特徴量を結合
            state_features = torch.cat([
                proprioceptive_state.flatten(),      # 自己受容状態
                text_features.flatten(),              # テキスト特徴量
                visual_features.flatten()             # 視覚特徴量
            ], dim=0)
            return state_features

    def generate_actions_from_noise(self, state_features: torch.Tensor, 
                                  latent_noise: torch.Tensor, 
                                  obs: Dict, task_desc: str) -> torch.Tensor:
        """
        潜在ノイズから行動チャンクを生成
        
        Args:
            state_features: VLMから抽出された状態特徴量
            latent_noise: 潜在ノイズ（単一ステップ）
            obs: 元の観測（画像等の再構築用）
            task_desc: タスク記述
        
        Returns:
            torch.Tensor: 生成された行動チャンク
        """
        with torch.no_grad():
            batch = self._prepare_batch(obs, task_desc)
            
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
            if hasattr(self.smolvla_policy.config, 'output_features') and 'action' in self.smolvla_policy.config.output_features:
                original_action_dim = self.smolvla_policy.config.output_features['action'].shape[0]
                actions = actions[:, :, :original_action_dim]
            else:
                # デフォルトのアクション次元を使用
                original_action_dim = min(self.max_action_dim, actions.shape[-1])
                actions = actions[:, :, :original_action_dim]
            
            # 正規化を解除
            actions = self.smolvla_policy.unnormalize_outputs({"action": actions})["action"]
            
            # Aloha環境の場合の変換
            if getattr(self.smolvla_policy.config, 'adapt_to_pi_aloha', False):
                actions = self.smolvla_policy._pi_aloha_encode_actions(actions)
            
            return actions.squeeze(0)  # バッチ次元を除去
    
    def _extract_text_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """SmolVLAのテキストエンコーダーから特徴量を抽出（ImageValueNetworkから移植）"""
        device = list(batch.values())[0].device if batch else self.device
        tasks = batch["task"]
        batch_size = 1  # DSRLでは通常バッチサイズは1
        
        # tasksがstrならリスト化
        if isinstance(tasks, str):
            tasks = [tasks]
        # tasksの長さがバッチサイズと異なる場合は複製
        if len(tasks) != batch_size:
            tasks = [tasks[0] for _ in range(batch_size)]
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]
        
        tokenized_prompt = self.smolvla_policy.language_tokenizer.__call__(
            tasks,
            padding=self.smolvla_policy.config.pad_language_to,
            padding_side="right",
            max_length=self.smolvla_policy.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_emb = self.smolvla_policy.model.vlm_with_expert.embed_language_tokens(lang_tokens)
        
        # 平均値プーリング
        pooled_features = lang_emb.mean(dim=1)
        return pooled_features.float()
    
    def _extract_vision_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """SmolVLAのビジョンエンコーダーから特徴量を抽出（ImageValueNetworkから移植）"""
        # 画像を取得
        images = []
        
        for key in ['observation.images.front', 'observation.images.side']:
            if key in batch:
                img = batch[key]
                if img.dim() == 4:  # [B, C, H, W]
                    images.append(img)
        
        # SmolVLAのビジョンエンコーダーを使用して特徴量を抽出
        all_features = []
        for img in images:
            # 画像を[-1, 1]の範囲に正規化（SmolVLAの要求に合わせる）
            if img.max() <= 1.0:
                img = img * 2.0 - 1.0
            
            # 画像埋め込み
            img_features = self.smolvla_policy.model.vlm_with_expert.embed_image(img)
            img_features = img_features.float()
            
            # 平均値プーリング
            pooled_features = img_features.mean(dim=1)
            all_features.append(pooled_features)
        
        if all_features:
            # 画像をconcat
            vision_features = torch.cat(all_features, dim=-1)
        else:
            # 画像がない場合はゼロベクトル
            vision_features = torch.zeros(1, self.visual_features_dim, device=self.device)
        
        return vision_features.float()
    
    def _prepare_batch(self, obs: Dict, task_desc: str) -> Dict[str, torch.Tensor]:
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
        
        # 画像情報 - GenesisEnvの観測形式をSmolVLAの期待する形式にマッピング
        # キーマッピングを修正：環境のキー -> SmolVLAの期待するキー
        image_mapping = {
            'observation.images.front': 'observation.images.front',
            'observation.images.side': 'observation.images.side',
        }
        
        for env_key, smolvla_key in image_mapping.items():
            if env_key in obs:
                img = obs[env_key]
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
                
                # バッチ次元を追加してSmolVLAの期待するキーで保存
                batch[smolvla_key] = img.to(self.device).unsqueeze(0)
        batch['task'] = task_desc
        
        return batch

def load_smolvla_model(model_path: Optional[str] = None, config_overrides: Optional[Dict] = None) -> SmolVLAPolicy:
    """
    SmolVLAモデルを読み込み
    
    Args:
        model_path: 事前学習済みモデルのパスまたはHubリポジトリ名
        config_overrides: 設定のオーバーライド
    
    Returns:
        SmolVLAPolicy: SmolVLAポリシー
    """
    config_overrides = config_overrides or {}
    
    try:
        # lerobot/common/policies/factory.pyの実装を参考にした読み込み
        from lerobot.common.policies.factory import get_policy_class
        
        # SmolVLAPolicyクラスを取得
        policy_cls = get_policy_class("smolvla")
        
        # 事前学習済みモデルを読み込み
        policy = policy_cls.from_pretrained(model_path)
        
        # コンフィグをオーバーライド
        for key, value in config_overrides.items():
            setattr(policy.config, key, value)
        
        logging.info(f"Loaded SmolVLA model from {model_path} with overrides.")
    
    except Exception as e:
        logging.warning(f"Could not load pre-trained model from '{model_path}'. Error: {e}. Creating a new one.")
        # プログラム終了
        exit(1)

    return policy

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
