"""
NoiseActionEnv - Tianshou compatible wrapper for DSRL

This environment wrapper:
1. Takes latent noise as actions from RL agent
2. Uses SmolVLAPolicy to convert noise to actual actions  
3. Executes actual actions in GenesisEnv
4. Returns rich state features as observations
"""

import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional
import logging
from pathlib import Path

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.genesis_env import GenesisEnv

class NoiseActionEnv(gym.Env):
    """
    Gym環境ラッパー：GenesisEnvとSmolVLAPolicyを組み合わせてDSRLを実現
    
    Action Space: ノイズベクトル (noise_dim,)
    Observation Space: リッチな状態特徴量 (total_state_dim,)
    """
    
    def __init__(self, genesis_env: GenesisEnv, smolvla_wrapper, chunk_size: int = 50, device: str = "cuda"):
        """
        Args:
            genesis_env: GenesisEnv instance
            smolvla_wrapper: SmolVLAWrapper instance from dsrl.py
            chunk_size: Action chunk size for SmolVLA
            device: Device for tensor operations
        """
        super().__init__()
        
        self.genesis_env = genesis_env
        self.smolvla_wrapper = smolvla_wrapper
        self.chunk_size = chunk_size
        self.device = device
        
        # Action space: ノイズ空間
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.smolvla_wrapper.noise_dim,), 
            dtype=np.float32
        )
        
        # Observation space: リッチな状態特徴量
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.smolvla_wrapper.total_state_dim,),
            dtype=np.float32
        )
        
        # Internal state
        self.current_obs = None
        self.current_state_features = None
        self.task_desc = None
        
        # For video recording
        self.frames = []
        self.record_video = False
        
        logging.info(f"NoiseActionEnv initialized: noise_dim={self.smolvla_wrapper.noise_dim}, "
                    f"state_dim={self.smolvla_wrapper.total_state_dim}, "
                    f"chunk_size={chunk_size}")
    
    def reset(self, seed=None, options=None):
        """環境をリセットし、初期状態特徴量を返す"""
        # GenesisEnvをリセット
        self.current_obs, info = self.genesis_env.reset(seed=seed, options=options)
        
        # タスク記述を取得
        self.task_desc = self.genesis_env.get_task_description()
        
        # SmolVLAで状態特徴量を抽出
        self.current_state_features = self.smolvla_wrapper.extract_state_features(
            self.current_obs, self.task_desc
        )
        
        # Video recording reset
        self.frames = []
        
        # tianshouのために状態特徴量をnumpy配列に変換
        state_features_np = self.current_state_features.cpu().numpy()
        
        return state_features_np, info
    
    def step(self, noise_action):
        """
        ノイズアクションを受け取り、SmolVLAで実際のアクションに変換して実行
        
        Args:
            noise_action: numpy array of shape (noise_dim,)
        
        Returns:
            next_state_features: numpy array of shape (total_state_dim,)
            reward: float
            terminated: bool
            truncated: bool
            info: dict
        """
        # numpy -> torch tensor
        if isinstance(noise_action, np.ndarray):
            noise_tensor = torch.from_numpy(noise_action).float().to(self.device)
        else:
            noise_tensor = noise_action.to(self.device)
        
        # Video frame recording
        if self.record_video:
            frame = self._render_frame()
            if frame is not None:
                self.frames.append(frame)
        
        # SmolVLAでアクションチャンクを生成
        try:
            action_chunk = self.smolvla_wrapper.generate_actions_from_noise(
                self.current_state_features, noise_tensor, self.current_obs, self.task_desc
            )
        except Exception as e:
            logging.warning(f"Failed to generate actions from noise: {e}")
            # フォールバック：ランダムアクション
            action_space = self.genesis_env.action_space
            if hasattr(action_space, 'sample'):
                fallback_action = action_space.sample()
                action_chunk = torch.from_numpy(fallback_action).float().to(self.device).unsqueeze(0)
            else:
                # Default action shape
                action_chunk = torch.zeros(1, 7, device=self.device)  # Assuming 7-DoF robot
        
        # アクションチャンクを逐次実行
        total_reward = 0.0
        done = False
        info = {}
        
        for action_idx in range(min(self.chunk_size, len(action_chunk))):
            if done:
                break
            
            action = action_chunk[action_idx]
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            # GenesisEnvでアクションを実行
            next_obs, reward, terminated, truncated, step_info = self.genesis_env.step(action)
            
            total_reward += reward
            done = terminated or truncated
            info.update(step_info)
            
            # Video frame recording
            if self.record_video and not done:
                frame = self._render_frame()
                if frame is not None:
                    self.frames.append(frame)
            
            if done:
                self.current_obs = next_obs
                break
            
            self.current_obs = next_obs
        
        # 新しい状態特徴量を抽出
        self.current_state_features = self.smolvla_wrapper.extract_state_features(
            self.current_obs, self.task_desc
        )
        
        # tianshouのために状態特徴量をnumpy配列に変換
        next_state_features_np = self.current_state_features.cpu().numpy()
        
        return next_state_features_np, total_reward, terminated, truncated, info
    
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
    
    def _render_frame(self):
        """動画用フレームをレンダリング"""
        try:
            frames = []
            image_keys = ['observation.images.front', 'observation.images.side']
            
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
                return combined
            
        except Exception as e:
            logging.warning(f"Failed to render frame: {e}")
        
        return None

    @property
    def max_episode_steps(self):
        """最大エピソードステップ数"""
        return getattr(self.genesis_env, '_max_episode_steps', 500)
