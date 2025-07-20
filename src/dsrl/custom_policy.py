import logging
from typing import Dict, Optional
import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import get_policy_class

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
        self.chunk_size = self.smolvla_policy.config.chunk_size
        self.noise_dim = self.smolvla_policy.config.max_action_dim
        # VLMの隠れ状態次元を取得
        self.vlm_hidden_size = self.smolvla_policy.model.vlm_with_expert.config.text_config.hidden_size
        # 1. 自己受容状態の次元
        self.proprioceptive_dim = self.smolvla_policy.config.max_state_dim
        # 2. テキスト特徴量の次元（ImageValueNetworkと同様）
        self.text_features_dim = 960
        # 3. 視覚特徴量の次元（ImageValueNetworkと同様）
        self.visual_features_dim = 960 * 2
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

    def generate_actions_from_noise(self, latent_noise: torch.Tensor, obs: Dict, task_desc: str) -> torch.Tensor:
        """
        潜在ノイズから行動チャンクを生成
        Args:
            latent_noise: 潜在ノイズ（単一ステップ）
            obs: 元の観測（画像等の再構築用）
            task_desc: タスク記述
        Returns:
            torch.Tensor: 生成された行動チャンク
        """
        with torch.no_grad():
            batch = self._prepare_batch(obs, task_desc)
            # 単一ステップのノイズをchunk_sizeにコピー
            if latent_noise.dim() == 1:
                latent_noise = latent_noise.unsqueeze(0)  # バッチ次元を追加
            # (batch_size, 1, noise_dim) -> (batch_size, chunk_size, noise_dim)
            noise_chunk = None
            # noise_chunk = latent_noise.unsqueeze(1).repeat(1, self.chunk_size, 1)
            actions = self.smolvla_policy._get_action_chunk(batch, noise_chunk)
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
        observation = {}
        for key in self.smolvla_policy.config.input_features:
            # make_sim_dataset.pyの観測キーに合わせてマッピング
            if key == "observation.state":
                data = obs["agent_pos"]
                tensor_data = torch.from_numpy(data).to(torch.float32)
                observation[key] = tensor_data.to(self.device).unsqueeze(0)
            elif key == "observation.images.front":
                img = obs["observation.images.front"]
                img = img.copy()  # 負のstride対策
                tensor_img = torch.from_numpy(img).to(torch.float32) / 255.0
                if tensor_img.ndim == 3 and tensor_img.shape[2] in [1, 3, 4]:
                    tensor_img = tensor_img.permute(2, 0, 1)
                elif tensor_img.ndim == 2:
                    tensor_img = tensor_img.unsqueeze(0)
                observation[key] = tensor_img.to(self.device).unsqueeze(0)
            elif key == "observation.images.side":
                img = obs["observation.images.side"]
                img = img.copy()  # 負のstride対策
                tensor_img = torch.from_numpy(img).to(torch.float32) / 255.0
                if tensor_img.ndim == 3 and tensor_img.shape[2] in [1, 3, 4]:
                    tensor_img = tensor_img.permute(2, 0, 1)
                elif tensor_img.ndim == 2:
                    tensor_img = tensor_img.unsqueeze(0)
                observation[key] = tensor_img.to(self.device).unsqueeze(0)
            else:
                print(f"Warning: Unsupported input feature '{key}'. Skipping.")
        observation["task"] = task_desc
        batch = self.smolvla_policy._prepare_batch(observation)
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
    policy_cls = get_policy_class("smolvla")
    policy = policy_cls.from_pretrained(model_path)
    for key, value in config_overrides.items():
        setattr(policy.config, key, value)
    logging.info(f"Loaded SmolVLA model from {model_path} with overrides.")
    return policy