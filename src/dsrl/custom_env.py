import gymnasium as gym
import numpy as np
import torch
import logging
import sys
import os
import cv2
from typing import Optional
import wandb
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from env.genesis_env import GenesisEnv
from src.dsrl.custom_policy import SmolVLAWrapper, load_smolvla_model

class BaseCustomEnv(gym.Env):
    """
    3つの環境クラスの共通機能を提供するベースクラス
    """
    def __init__(self, config: dict):
        super().__init__()
        self.genesis_env = GenesisEnv(
            task='simple_pick',
            observation_height=config['observation_height'],
            observation_width=config['observation_width'],
            show_viewer=config.get('show_viewer', False)
        )
        self.device = config['device']
        # For video recording
        self.frames = []
        self.record_video = False
        self.reward = 0.0

    def render(self, mode="rgb_array"):
        """環境の描画"""
        return self.genesis_env.render()

    def close(self):
        """環境を閉じる"""
        self.genesis_env.close()

    def get_task_description(self):
        """タスク記述を取得"""
        return self.genesis_env.get_task_description()

    def start_video_recording(self):
        """動画記録を開始"""
        self.record_video = True
        self.frames = []

    def stop_video_recording(self):
        """動画記録を停止"""
        self.record_video = False
        return self.frames

    def render_frame(self) -> Optional[np.ndarray]:
        """現在の観測に報酬とタスク情報を追加してフレームをレンダリング"""
        task_desc = self.get_task_description()
        frames = []
        image_keys = ['observation.images.front', 'observation.images.side']
        # current_obsが存在する場合のみフレームを生成
        if self.current_obs is not None:
            for key in image_keys:
                if key in self.current_obs:
                    img = self.current_obs[key]
                    if isinstance(img, torch.Tensor):
                        img = img.cpu().numpy()
                    if isinstance(img, np.ndarray):
                        img = img.copy()
                    # 画像の次元を調整
                    if img.ndim == 3 and img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
                    # 値の範囲を[0,255]に正規化
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    frames.append(img)
        if frames:
            # 複数の画像を横に結合
            combined = np.concatenate(frames, axis=1)
            # 報酬を表示
            if self.reward is not None:
                cv2.putText(
                    combined, f"Reward: {self.reward:.2f}",
                    (10, combined.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                )
            # タスク記述を表示
            if task_desc is not None:
                cv2.putText(
                    combined,
                    task_desc,
                    (10, combined.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
            return combined
        return None

    def upload_video_to_wandb(self, frames: list) -> None:
        """動画をWandBにアップロード"""
        if not frames:
            return
        # フレームを(T, H, W, C)形式に変換
        video_array = np.stack(frames, axis=0)  # (T, H, W, C)
        # 値の範囲を[0, 255]に正規化し、uint8に変換
        if video_array.max() <= 1.0:
            video_array = (video_array * 255).astype(np.uint8)
        else:
            video_array = video_array.astype(np.uint8)
        # THWC -> TCHW
        video_array = np.transpose(video_array, (0, 3, 1, 2))
        wandb.log({
            f"videos/training_video": wandb.Video(video_array, fps=30, format="mp4")
        })

    @property
    def max_episode_steps(self):
        return self.genesis_env._max_episode_steps

class StateObsEnv(BaseCustomEnv):
    """
    観測として画像ではなく状態特徴量を返すGym環境ラッパー
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.action_space = self.genesis_env.action_space
        # observationとして目標のboxの座標とjointの位置・速度を返す
        # joint 9自由度 * 2 (位置, 速度) + boxの座標3次元 = 21次元
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            # shape=(3,), dtype=np.float32
            # shape=(12,), dtype=np.float32
            shape=(21,), dtype=np.float32
        )
        self.old_pos = None
        self.current_obs = None

    def reset(self, seed=None, options=None):
        """環境をリセットし、初期状態特徴量を返す"""
        self.old_pos = None
        self.current_obs, info = self.genesis_env.reset(seed=seed, options=options)
        new_obs = self.make_obs(self.current_obs)
        # Video recording reset
        self.frames = []
        self.reward = 0.0
        if self.record_video:
            frame = self.render_frame()
            if frame is not None:
                self.frames.append(frame)
        return new_obs, info

    def make_obs(self, obs):
        jooint_pos = obs['agent_pos']
        joint_vel = np.zeros_like(jooint_pos) if self.old_pos is None else (jooint_pos - self.old_pos)
        if self.genesis_env._env.color == "red":
            target_pos = self.genesis_env._env.cubeA.get_pos().cpu().numpy()
        elif self.genesis_env._env.color == "blue":
            target_pos = self.genesis_env._env.cubeB.get_pos().cpu().numpy()
        elif self.genesis_env._env.color == "green":
            target_pos = self.genesis_env._env.cubeC.get_pos().cpu().numpy()
        else:
            raise ValueError(f"Unknown color: {self.genesis_env._env.color}")
        target_pos -= self.genesis_env._env.eef.get_pos().cpu().numpy() # エンドエフェクタからの相対位置に変換
        # new_obs = target_pos
        # new_obs = np.concatenate([jooint_pos, target_pos])
        new_obs = np.concatenate([jooint_pos, joint_vel, target_pos])
        self.old_pos = jooint_pos.copy()
        return new_obs

    def step(self, action):
        """アクションを実行し、次の状態特徴量を返す"""
        self.current_obs, self.reward, terminated, truncated, info = self.genesis_env.step(action)
        new_obs = self.make_obs(self.current_obs)
        if self.record_video and not (terminated or truncated):
            frame = self.render_frame()
            if frame is not None:
                self.frames.append(frame)
        return new_obs, self.reward, terminated, truncated, info

class NoiseActionEnv(BaseCustomEnv):
    """
    Gym環境ラッパー：GenesisEnvとSmolVLAPolicyを組み合わせてDSRLを実現
    Action Space: ノイズベクトル (noise_dim,)
    Observation Space: StateObsEnvと同じ21次元の状態ベクトル
    """
    def __init__(self, config: dict):
        super().__init__(config)
        smolvla_policy = load_smolvla_model(
            config['pretrained_model_path'],
            config['smolvla_config_overrides']
        )
        self.smolvla_wrapper = SmolVLAWrapper(smolvla_policy, config['device'])
        # Action space: ノイズ空間
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.smolvla_wrapper.noise_dim,),
            dtype=np.float32
        )
        # Observation space: StateObsEnvと同じ21次元の状態ベクトル
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(21,), dtype=np.float32
        )
        self.n_action_steps = self.smolvla_wrapper.smolvla_policy.config.n_action_steps
        # Internal state
        self.current_obs = None
        self.task_desc = None
        self.old_pos = None
        logging.info(f"NoiseActionEnv initialized: noise_dim={self.smolvla_wrapper.noise_dim}, "
                    f"obs_dim=21, chunk_size={self.smolvla_wrapper.chunk_size}")

    def reset(self, seed=None, options=None):
        """環境をリセットし、初期状態特徴量を返す"""
        self.current_obs, info = self.genesis_env.reset(seed=seed, options=options)
        self.task_desc = self.genesis_env.get_task_description()
        # StateObsEnv形式の観測を生成
        self.old_pos = None
        state_obs = self.make_obs(self.current_obs)
        self.frames = []
        self.reward = 0.0
        if self.record_video:
            frame = self.render_frame()
            if frame is not None:
                self.frames.append(frame)
        return state_obs, info

    def make_obs(self, obs):
        """StateObsEnvと同じ形式の観測を生成"""
        jooint_pos = obs['agent_pos']
        joint_vel = np.zeros_like(jooint_pos) if self.old_pos is None else (jooint_pos - self.old_pos)
        if self.genesis_env._env.color == "red":
            target_pos = self.genesis_env._env.cubeA.get_pos().cpu().numpy()
        elif self.genesis_env._env.color == "blue":
            target_pos = self.genesis_env._env.cubeB.get_pos().cpu().numpy()
        elif self.genesis_env._env.color == "green":
            target_pos = self.genesis_env._env.cubeC.get_pos().cpu().numpy()
        else:
            raise ValueError(f"Unknown color: {self.genesis_env._env.color}")
        target_pos -= self.genesis_env._env.eef.get_pos().cpu().numpy() # エンドエフェクタからの相対位置に変換
        new_obs = np.concatenate([jooint_pos, joint_vel, target_pos])
        self.old_pos = jooint_pos.copy()
        return new_obs

    def step(self, noise_action):
        noise_tensor = torch.from_numpy(noise_action).float().to(self.device)
        action_chunk = self.smolvla_wrapper.generate_actions_from_noise(
            noise_tensor, self.current_obs, self.task_desc
        )
        # アクションチャンクを逐次実行
        total_reward = 0.0
        done = False
        info = {}
        for i in range(self.n_action_steps):
            action = action_chunk[i].cpu().numpy()
            # GenesisEnvでアクションを実行
            self.current_obs, self.reward, terminated, truncated, step_info = self.genesis_env.step(action)
            total_reward += self.reward
            done = terminated or truncated
            info.update(step_info)
            # Video frame recording
            if self.record_video and not done:
                frame = self.render_frame()
                if frame is not None:
                    self.frames.append(frame)
            if done:
                break
        # StateObsEnv形式の観測を生成
        state_obs = self.make_obs(self.current_obs)
        reward = total_reward / self.n_action_steps
        return state_obs, reward, terminated, truncated, info

class NoiseActionVisualEnv(BaseCustomEnv):
    """
    Gym環境ラッパー：GenesisEnvとSmolVLAPolicyを組み合わせてDSRLを実現
    Action Space: ノイズベクトル (noise_dim,)
    Observation Space: リッチな状態特徴量 (total_state_dim,)
    """
    def __init__(self, config: dict):
        super().__init__(config)
        smolvla_policy = load_smolvla_model(
            config['pretrained_model_path'],
            config['smolvla_config_overrides']
        )
        self.smolvla_wrapper = SmolVLAWrapper(smolvla_policy, config['device'])
        # Action space: ノイズ空間
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.smolvla_wrapper.noise_dim,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.smolvla_wrapper.total_state_dim,),
            dtype=np.float32
        )
        self.n_action_steps = self.smolvla_wrapper.smolvla_policy.config.n_action_steps
        # Internal state
        self.current_obs = None
        self.task_desc = None
        logging.info(f"NoiseActionVisualEnv initialized: noise_dim={self.smolvla_wrapper.noise_dim}, "
                    f"state_dim={self.smolvla_wrapper.total_state_dim}, "
                    f"chunk_size={self.smolvla_wrapper.chunk_size}")

    def reset(self, seed=None, options=None):
        """環境をリセットし、初期状態特徴量を返す"""
        self.current_obs, info = self.genesis_env.reset(seed=seed, options=options)
        self.task_desc = self.genesis_env.get_task_description()
        self.frames = []
        self.reward = 0.0
        # SmolVLAWrapperのextract_featuresを使用して統合特徴量を取得
        state_features = self.smolvla_wrapper.extract_features(self.current_obs, self.task_desc)
        # 特徴量にタスク情報を追加
        task_info = self.get_task_info()
        state_features = torch.cat(
            [state_features, torch.tensor([task_info], device='cpu', dtype=torch.float32)],
            dim=0
        )
        if self.record_video:
            frame = self.render_frame()
            if frame is not None:
                self.frames.append(frame)
        return state_features, info

    def step(self, noise_action):
        noise_tensor = torch.from_numpy(noise_action).float().to(self.device)
        # SmolVLAでアクションチャンクを生成
        action_chunk = self.smolvla_wrapper.generate_actions_from_noise(
            noise_tensor, self.current_obs, self.task_desc
        )
        # アクションチャンクを逐次実行
        total_reward = 0.0
        done = False
        info = {}
        for i in range(self.n_action_steps):
            action = action_chunk[i].cpu().numpy()
            # GenesisEnvでアクションを実行
            self.current_obs, self.reward, terminated, truncated, step_info = self.genesis_env.step(action)
            total_reward += self.reward
            done = terminated or truncated
            info.update(step_info)
            # Video frame recording
            if self.record_video and not done:
                frame = self.render_frame()
                if frame is not None:
                    self.frames.append(frame)
            if done:
                break
        # SmolVLAWrapperのextract_featuresを使用して統合特徴量を取得
        state_features = self.smolvla_wrapper.extract_features(self.current_obs, self.task_desc)
        # 特徴量にタスク情報を追加
        task_info = self.get_task_info()
        state_features = torch.cat(
            [state_features, torch.tensor([task_info], device='cpu', dtype=torch.float32)],
            dim=0
        )
        reward = total_reward / self.n_action_steps
        return state_features, reward, terminated, truncated, info

    def get_task_info(self):
        if 'green' in self.task_desc:
            return 0.0
        elif 'red' in self.task_desc:
            return 0.5
        elif 'blue' in self.task_desc:
            return 1.0
        else:
            return 0.0
