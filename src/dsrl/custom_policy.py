import logging
from typing import Dict, Optional, Union
import torch
import numpy as np
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
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
        self.proprioceptive_dim = self.smolvla_policy.config.max_state_dim
        self.vlm_dim = 720
        # 画像サイズを計算（512x512x3の画像2枚）
        self.img_height = 512
        self.img_width = 512
        self.img_channels = 3
        self.front_img_size = self.img_height * self.img_width * self.img_channels
        self.side_img_size = self.img_height * self.img_width * self.img_channels
        self.total_state_dim = self.front_img_size + self.side_img_size + self.vlm_dim + self.proprioceptive_dim
        self.actions = torch.zeros((1, self.chunk_size, self.noise_dim), device=self.device)

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
            batch.pop("action", None)  # 既存のactionを削除
            # 単一ステップのノイズをchunk_sizeにコピー
            if latent_noise.dim() == 1:
                latent_noise = latent_noise.unsqueeze(0)  # バッチ次元を追加
            # (batch_size, 1, noise_dim) -> (batch_size, chunk_size, noise_dim)
            noise_chunk = latent_noise.unsqueeze(1).repeat(1, self.chunk_size, 1)
            actions = self.smolvla_policy._get_action_chunk(batch, noise_chunk)
            self.actions = actions.to(self.device)  # デバイスに転送
            return actions.squeeze(0)  # バッチ次元を除去

    def extract_features(self, obs: Dict, task_desc: str) -> torch.Tensor:
        """
        Args:
            obs: 環境からの観測
            task_desc: タスク記述
        Returns:
            torch.Tensor: 統合された状態特徴量（画像データ、VLM特徴量、自己受容状態を含む）
        """
        with torch.no_grad():
            batch = self._prepare_batch(obs, task_desc)
            # 1. 自己受容状態（proprioceptive state）の取得
            proprioceptive_state = self.smolvla_policy.prepare_state(batch)  # (1, state_dim)
            # 2. VLM特徴量の抽出
            vlm_features = self._extract_vlm_features(batch)
            # 3. 画像データの取得（平坦化）
            front_img = batch["observation.images.front"].cpu().numpy()[0]  # (3, H, W)
            side_img = batch["observation.images.side"].cpu().numpy()[0]   # (3, H, W)
            # 画像を平坦化
            front_img_flat = front_img.flatten()
            side_img_flat = side_img.flatten()
            # 画像データ（平坦化）+ VLM特徴量 + 自己受容状態
            front_img_tensor = torch.from_numpy(front_img_flat).float().unsqueeze(0)
            side_img_tensor = torch.from_numpy(side_img_flat).float().unsqueeze(0)
            state_features = torch.cat([
                front_img_tensor,
                side_img_tensor,
                vlm_features.cpu(),
                proprioceptive_state.cpu()
            ], dim=1)
            return state_features.squeeze(0)  # バッチ次元を除去

    def _extract_vlm_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """SmolVLAのVLMの最終トークンから特徴量を取得"""
        images, img_masks = self.smolvla_policy.prepare_images(batch)
        state = self.smolvla_policy.prepare_state(batch)
        lang_tokens, lang_masks = self.smolvla_policy.prepare_language(batch)
        actions = self.smolvla_policy.prepare_action(batch)
        noise = self.smolvla_policy.model.sample_noise(actions.shape, actions.device)
        time = self.smolvla_policy.model.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.smolvla_policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.smolvla_policy.model.embed_suffix(x_t, time)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.smolvla_policy.model.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -1, :]  # 最終トークンの出力を取得
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    def _extract_vision_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """SmolVLAのビジョンエンコーダーから特徴量を抽出"""
        images, _ = self.smolvla_policy.prepare_images(batch)
        features = []
        for img in images:
            image_hidden_states = self.smolvla_policy.model.vlm_with_expert.get_vlm_model().vision_model(
                pixel_values=img.to(dtype=self.smolvla_policy.model.vlm_with_expert.get_vlm_model().vision_model.dtype),
                patch_attention_mask=None,
            ).last_hidden_state
            # image_hidden_states shape: [1, 1024, 768]
            resampled = self.smolvla_policy.model.vlm_with_expert.get_vlm_model().connector(image_hidden_states)
            # resampled shape: [1, 64, 960]
            feature = resampled.mean(dim=1)
            features.append(feature)
        return torch.cat(features, dim=1).to(dtype=torch.float32)

    def _prepare_batch(self, obs: Dict, task_desc: str) -> Dict[str, torch.Tensor]:
        """観測をSmolVLA用のバッチ形式に変換"""
        def prepare_image(img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
            if isinstance(img, np.ndarray):
                img = img.copy()  # 負のstride対策
                tensor_img = torch.from_numpy(img).to(torch.float32)
            elif isinstance(img, torch.Tensor):
                tensor_img = img.to(torch.float32)
            if tensor_img.max() > 1.0:
                tensor_img /= 255.0
            if tensor_img.ndim == 3 and tensor_img.shape[2] in [1, 3, 4]:
                tensor_img = tensor_img.permute(2, 0, 1)
            elif tensor_img.ndim == 2:
                tensor_img = tensor_img.unsqueeze(0)
            return tensor_img.to(self.device).unsqueeze(0)

        observation = {}
        for key in self.smolvla_policy.config.input_features:
            # make_sim_dataset.pyの観測キーに合わせてマッピング
            if key == "observation.state":
                if "agent_pos" in obs:
                    data = obs["agent_pos"]
                elif "observation.state" in obs:
                    data = obs["observation.state"]
                # ndarrayならTensorに変換
                if isinstance(data, np.ndarray):
                    tensor_data = torch.from_numpy(data).to(torch.float32)
                elif isinstance(data, torch.Tensor):
                    tensor_data = data.to(torch.float32)
                observation[key] = tensor_data.to(self.device).unsqueeze(0)
            elif key == "observation.images.front":
                img = obs["observation.images.front"]
                observation[key] = prepare_image(img)
            elif key == "observation.images.side":
                img = obs["observation.images.side"]
                observation[key] = prepare_image(img)
            elif key == "observation.images.eef":
                img = obs["observation.images.eef"]
                observation[key] = prepare_image(img)
            else:
                print(f"Warning: Unsupported input feature '{key}'. Skipping.")
        observation["task"] = task_desc
        observation["action"] = self.actions
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
