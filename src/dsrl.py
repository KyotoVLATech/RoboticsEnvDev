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
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import wandb

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.genesis_env import GenesisEnv
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from src.make_sim_dataset import task_description
from src.dsrl_agent import DSRLAgent, DSRLNA, DSRLSAC, DSRLExperience

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
            task: タスク記述
            env: 環境のインスタンス（動的なタスク記述生成用）
        
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
            task: タスク記述
            env: 環境のインスタンス（動的なタスク記述生成用）
        
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
        
        # 動画記録用
        self.video_frames = []
        
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
            
            # 評価と動画記録
            if episode % self.config.get('eval_freq', 50) == 0:
                self._run_evaluation(episode)
            
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
        if self.config['task'] in task_description:
            task_desc = task_description[self.config['task']]
        elif self.config['task'] == 'simple_pick':
            try:
                task_desc = f"Pick up a {self.env._env.color} cube."
            except AttributeError:
                task_desc = "Pick up a cube."
        else:
            task_desc = self.config['task']
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
            current_state_features = self.smolvla_wrapper.extract_state_features(obs, task_desc)

            # DSRLエージェントで潜在ノイズを選択
            deterministic = self.config.get('deterministic_eval', False) and \
                          self.episode_count % self.config.get('eval_freq', 50) == 0
            latent_noise = self.dsrl_agent.select_noise(current_state_features, deterministic=deterministic)
            
            # SmolVLAで行動チャンクを生成
            action_chunk = self.smolvla_wrapper.generate_actions_from_noise(
                current_state_features, latent_noise, obs, task_desc
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
                    obs = next_obs  # 最終観測を更新
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
    
    def _run_evaluation(self, episode: int) -> None:
        """評価エピソードを実行し、動画を記録"""
        self.logger.info(f"Running evaluation at episode {episode}")
        
        # 動画記録フラグを有効にする
        record_video = self.config.get('record_video', True)
        video_freq = self.config.get('video_freq', 10)
        should_record = record_video and episode % video_freq == 0
        
        # 動画フレームをリセット
        if should_record:
            self.video_frames = []
        
        # 評価エピソードを実行
        eval_rewards = []
        eval_successes = []
        num_eval_episodes = self.config.get('num_eval_episodes', 3)
        
        for eval_ep in range(num_eval_episodes):
            episode_reward, episode_length, success, frames = self._run_single_evaluation_episode(should_record)
            eval_rewards.append(episode_reward)
            eval_successes.append(float(success))
            
            # 最初の評価エピソードの動画フレームを記録
            if should_record and eval_ep == 0 and frames:
                self.video_frames.extend(frames)
        
        # 評価結果をログ
        avg_eval_reward = np.mean(eval_rewards)
        avg_eval_success = np.mean(eval_successes)
        
        self.logger.info(
            f"Evaluation at episode {episode}: "
            f"avg_reward={avg_eval_reward:.3f}, avg_success={avg_eval_success:.3f}"
        )
        
        # WandBにログ
        if self.config.get('use_wandb', True):
            wandb.log({
                'eval/avg_reward': avg_eval_reward,
                'eval/avg_success': avg_eval_success,
                'eval/episode': episode
            })
        
        # 動画をWandBにアップロード
        if should_record and self.video_frames:
            self._upload_video(episode)
    
    def _run_single_evaluation_episode(self, record_video: bool = False) -> Tuple[float, int, bool, Optional[List]]:
        """単一の評価エピソードを実行"""
        obs, info = self.env.reset()
        if self.config['task'] in task_description:
            task_desc = task_description[self.config['task']]
        elif self.config['task'] == 'simple_pick':
            try:
                task_desc = f"Pick up a {self.env._env.color} cube."
            except AttributeError:
                task_desc = "Pick up a cube."
        else:
            task_desc = self.config['task']
        done = False
        episode_reward = 0.0
        episode_length = 0
        frames = [] if record_video else None
        
        while not done and episode_length < self.config.get('max_episode_length', 500):
            # 状態特徴量を抽出
            state_features = self.smolvla_wrapper.extract_state_features(obs, task_desc)
            # 決定論的にノイズを選択（評価時）
            latent_noise = self.dsrl_agent.select_noise(state_features, deterministic=True)
            # SmolVLAで行動チャンクを生成
            action_chunk = self.smolvla_wrapper.generate_actions_from_noise(
                state_features, latent_noise, obs, task_desc
            )
            # Action chunkを逐次実行
            for action_idx in range(min(self.config.get('chunk_size', 50), len(action_chunk))):
                # 各ステップで動画フレームを記録
                if record_video:
                    frame = self._render_frame(obs)
                    if frame is not None:
                        frames.append(frame)
                
                action = action_chunk[action_idx]
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                # 終了条件をチェック
                done = terminated or truncated
                if done:
                    break
            
            # アクションチャンク実行後も終了条件をチェック
            if done:
                break
        
        success = info.get('is_success', False)
        return episode_reward, episode_length, success, frames
    
    def _render_frame(self, obs: Dict) -> Optional[np.ndarray]:
        """観測から動画フレームをレンダリング"""
        try:
            frames = []
            image_keys = ['observation.images.front', 'observation.images.side']
            
            for key in image_keys:
                if key in obs:
                    img = obs[key]
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
            self.logger.warning(f"Failed to render frame: {e}")
        
        return None
    
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
                f"eval_video/video": wandb.Video(video_array, fps=30, format="mp4")
            })
            
            self.logger.info(f"Uploaded evaluation video for episode {episode}")
            
        except Exception as e:
            self.logger.warning(f"Failed to upload video: {e}")

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
        'hidden_dim': 128,
        'learning_rate': 0.0003,
        'gamma': 0.999,
        'tau': 0.005,
        'target_update_freq': 2,
        
        # 学習設定
        'batch_size': 256,
        'replay_buffer_size': 100000,
        'min_replay_size': 1000,
        'updates_per_episode': 1,
        
        # ログ・保存設定
        'use_wandb': True,
        'wandb_project': 'dsrl-smolvla',
        'wandb_run_name': None,  # Noneの場合は自動生成
        'log_freq': 10,
        'save_freq': 50,
        'eval_freq': 50,
        'deterministic_eval': False,
        'checkpoint_dir': 'outputs/train/dsrl_checkpoints',
        
        # 動画記録設定
        'record_video': True,
        'video_freq': 50,  # 動画記録頻度（エピソード毎）
        'num_eval_episodes': 1,  # 評価時のエピソード数
        
        # SmolVLA設定
        'pretrained_model_path': "outputs/train/smolvla_test_0/checkpoints/last/pretrained_model",  # 事前学習済みモデルのパス
        'smolvla_config_overrides': {
            'n_action_steps': 20,
        }
    }
    
    # コマンドライン引数の処理
    import argparse
    parser = argparse.ArgumentParser(description="Train or evaluate a DSRL agent on SmolVLA.")
    parser.add_argument('--eval', action='store_true', help='Run in evaluation mode.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the agent checkpoint for evaluation (e.g., outputs/train/dsrl_checkpoints/episode_2000/dsrl_agent.pth).')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes for evaluation.')
    
    # Allow overriding config values from command line
    parser.add_argument('--task', type=str, default=config['task'], help=f"Task to run. Default: {config['task']}")
    parser.add_argument('--algorithm', type=str, default=config['algorithm'], help=f"DSRL algorithm ('NA' or 'SAC'). Default: {config['algorithm']}")
    parser.add_argument('--total_episodes', type=int, default=config['total_episodes'], help=f"Total episodes for training. Default: {config['total_episodes']}")
    parser.add_argument('--pretrained_model_path', type=str, default=config['pretrained_model_path'], help=f"Path to pretrained SmolVLA. Default: {config['pretrained_model_path']}")
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging.')
    parser.add_argument('--show_viewer', action='store_true', help='Show environment viewer during training.')

    args = parser.parse_args()

    # Update config with command line arguments
    config['task'] = args.task
    config['algorithm'] = args.algorithm
    config['total_episodes'] = args.total_episodes
    config['pretrained_model_path'] = args.pretrained_model_path
    if args.no_wandb: config['use_wandb'] = False
    if args.show_viewer: config['show_viewer'] = True

    if args.eval:
        if not args.checkpoint_path:
            parser.error("--checkpoint_path is required for evaluation.")
        
        # In eval mode, try to load config from the checkpoint for consistency
        eval_config = config.copy()
        try:
            ckpt = torch.load(args.checkpoint_path, map_location='cpu')
            if 'config' in ckpt:
                # Overwrite the base config with the one from the checkpoint
                # This ensures that model parameters (hidden_dim, etc.) match
                eval_config = ckpt['config']
                # But allow command-line overrides for things we might want to change at eval time
                eval_config['task'] = args.task # Use task from command line
            else:
                print("Warning: No config found in checkpoint. Using default config.")

        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
            return
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return

        evaluate_dsrl_model(args.checkpoint_path, eval_config, num_episodes=args.num_episodes)
        return
    
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
    action_dim = smolvla_wrapper.max_action_dim
    
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
    action_dim = smolvla_wrapper.max_action_dim
    
    dsrl_agent = create_dsrl_agent(
        config['algorithm'], state_dim, noise_dim, action_dim, config, device
    )
    dsrl_agent.load_checkpoint(checkpoint_path)
    
    # 評価実行
    total_rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        if config['task'] in task_description:
            task_desc = task_description[config['task']]
        elif config['task'] == 'simple_pick':
            try:
                task_desc = f"Pick up a {env._env.color} cube."
            except AttributeError:
                task_desc = "Pick up a cube."
        else:
            task_desc = config['task']
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done and episode_length < config.get('max_episode_length', 500):
            # 状態特徴量を抽出
            state_features = smolvla_wrapper.extract_state_features(obs, task_desc)

            # 決定論的にノイズを選択
            latent_noise = dsrl_agent.select_noise(state_features, deterministic=True)
            
            # 行動チャンクを生成
            action_chunk = smolvla_wrapper.generate_actions_from_noise(
                state_features, latent_noise, obs, task_desc
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
