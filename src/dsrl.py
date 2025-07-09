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
        
        # 2. VLM最終トークン特徴量の次元（論文では2048次元）
        self.vlm_final_token_dim = self.vlm_hidden_size
        
        # 3. 視覚特徴量の次元（SmolVLAの実際の画像エンコーダー出力次元に基づく）
        # embed_imageメソッドから実際の次元を取得
        self.visual_features_dim = self._get_visual_features_dim()
        
        # 総合的な状態特徴量の次元
        self.total_state_dim = (
            self.proprioceptive_dim + 
            self.vlm_final_token_dim + 
            self.visual_features_dim
        )
    
    def _get_visual_features_dim(self) -> int:
        """
        SmolVLAの画像エンコーダーから実際の視覚特徴量次元を取得
        
        Returns:
            int: 視覚特徴量の次元（複数カメラ分を考慮）
        """
        try:
            # ダミー画像を作成してembed_imageメソッドで実際の出力次元を確認
            dummy_img = torch.randn(1, 3, 224, 224, device=self.device)
            
            # SmolVLAの正規化処理を適用
            dummy_img = dummy_img * 2.0 - 1.0
            
            with torch.no_grad():
                # embed_imageで実際の特徴量を取得
                img_features = self.smolvla_policy.model.vlm_with_expert.embed_image(dummy_img)
                
                # 特徴量の次元を取得
                if img_features.dim() == 3:  # (batch, seq_len, hidden_dim)
                    single_cam_dim = img_features.shape[-1]
                elif img_features.dim() == 4:  # (batch, height, width, hidden_dim)
                    single_cam_dim = img_features.shape[-1]
                else:
                    single_cam_dim = img_features.shape[-1]
                
                # 2カメラ分の次元（front, side）
                total_visual_dim = single_cam_dim * 2
                
                logging.info(f"Detected visual feature dimensions: {single_cam_dim} per camera, {total_visual_dim} total")
                return total_visual_dim
                
        except Exception as e:
            logging.warning(f"Failed to detect visual feature dimensions: {e}. Using fallback value.")
            # フォールバック：SmolVLAの隠れ状態次元を使用
            return self.vlm_hidden_size
    
    def extract_state_features(self, obs: Dict, task_desc: str) -> torch.Tensor:
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
            task_desc: タスク記述
        
        Returns:
            torch.Tensor: 論文に基づく状態特徴量
        """
        with torch.no_grad():
            batch = self._prepare_batch(obs, task_desc)
            
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
            
            # 3. 視覚特徴量の取得（SmolVLAの実装に基づく）
            # SmolVLAのembed_imageメソッドを使用して豊富な視覚特徴量を取得
            visual_features_list = []
            image_keys = ['observation.images.front', 'observation.images.side']
            
            for key in image_keys:
                if key in batch:
                    img_tensor = batch[key]  # (1, C, H, W)
                    
                    # SmolVLAの画像エンコーダーを使用して特徴量を抽出
                    # prepare_imagesと同様の前処理を適用
                    img_processed = img_tensor.clone()
                    
                    # SmolVLAの正規化: [0,1] -> [-1,1] (SigLIP用)
                    if img_processed.max() <= 1.0:
                        img_processed = img_processed * 2.0 - 1.0
                    
                    # SmolVLAのembed_imageメソッドを使用
                    img_features = self.smolvla_policy.model.vlm_with_expert.embed_image(img_processed)
                    
                    # 画像特徴量の次元を削減（Global Average Pooling）
                    if img_features.dim() == 3:  # (batch, seq_len, hidden_dim)
                        img_features = img_features.mean(dim=1)  # (batch, hidden_dim)
                    elif img_features.dim() == 4:  # (batch, height, width, hidden_dim)
                        img_features = img_features.mean(dim=(1, 2))  # (batch, hidden_dim)
                    
                    visual_features_list.append(img_features)
            
            if visual_features_list:
                # 複数画像の特徴量を結合
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
