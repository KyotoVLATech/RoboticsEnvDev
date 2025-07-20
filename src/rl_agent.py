import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.distributions as dist
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.constants import ACTION
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
    """価値関数ネットワーク（gym環境用に拡張）"""
    def __init__(self, state_dim: int, hidden_dim: int = 256, normalize_input: bool = True):
        super().__init__()
        self.normalize_input = normalize_input
        
        # 入力正規化用のパラメータ
        if self.normalize_input:
            self.input_mean = nn.Parameter(torch.zeros(state_dim), requires_grad=False)
            self.input_std = nn.Parameter(torch.ones(state_dim), requires_grad=False)
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # 入力正規化
        if self.normalize_input:
            state = (state - self.input_mean) / (self.input_std + 1e-8)
        
        return self.network(state).squeeze(-1)
    
    def update_normalization(self, states: torch.Tensor):
        """観測の正規化パラメータを更新"""
        if self.normalize_input and states.numel() > 0:
            self.input_mean.data = states.mean(dim=0)
            self.input_std.data = states.std(dim=0)

class ImageValueNetwork(nn.Module):
    """
    画像特徴量を使った価値関数ネットワーク
    現在の実装ではVLMのトークナイザを流用しているが，最終的にはVLMのActionExpertへの出力を取ってきて入力としたい．
    attentionプーリングの実装を検討
    もしSmolVLA側に[CLS]トークンがあればそれを使う
    """
    def __init__(self, smolvla_policy: SmolVLAPolicy, hidden_dim: int = 256):
        super().__init__()
        self.smolvla_policy = smolvla_policy
        self._vision_feature_dim = 960*3
        # 価値関数のヘッド
        self.value_head = nn.Sequential(
            nn.Linear(self._vision_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """観測から価値を予測"""
        with torch.no_grad():
            # SmolVLAのビジョンエンコーダーを使って特徴量を抽出
            vision_features = self._extract_vision_features(obs_batch)
            # SmolVLAのテキストエンコーダーを使ってタスク記述をエンコード
            text_features = self._extract_text_features(obs_batch)
            features = torch.cat([vision_features, text_features], dim=-1)
        if features.device != self.value_head[0].weight.device:
            self.value_head.to(features.device)
        value = self.value_head(features)
        return value.squeeze(-1)
    
    def _extract_text_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """SmolVLAのテキストエンコーダーから特徴量を抽出"""
        device = batch['observation.images.front'].device
        tasks = batch["task"]
        batch_size = batch['observation.images.front'].shape[0]
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
        # ([1, 48, 960])
        # 平均値プーリング
        pooled_features = lang_emb.mean(dim=1)
        return pooled_features.float()
    
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
        
        # SmolVLAのビジョンエンコーダーを使用して特徴量を抽出
        # VLMWithExpertモデルのembed_imageメソッドを使用
        all_features = []
        for img in images:
            # 画像を[-1, 1]の範囲に正規化（SmolVLAの要求に合わせる）
            if img.max() <= 1.0:
                img = img * 2.0 - 1.0
            # 画像埋め込み
            img_features = self.smolvla_policy.model.vlm_with_expert.embed_image(img)
            img_features = img_features.float()
            # ([B, 64, 960])
            # 平均値プーリング
            pooled_features = img_features.mean(dim=1)
            all_features.append(pooled_features)
        
        # 画像をconcat
        vision_features = torch.cat(all_features, dim=-1)  # ([B, 1920])
        return vision_features.float()


class SmolVLAPolicyWrapper(nn.Module):
    def __init__(self, smolvla_policy: SmolVLAPolicy, action_dim: int, initial_std: float = 0.1, min_std: float = 0.05):
        super().__init__()
        self.smolvla_policy = smolvla_policy
        # PPO用の標準偏差パラメータを追加
        self.log_std = nn.Parameter(torch.full((action_dim,), np.log(initial_std)))
        self.min_std = min_std  # 最小標準偏差
        
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
        std = torch.exp(self.log_std)
        # mean_actionの形状に合わせてstdを調整
        if mean_action.dim() == 2:
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
        
        log_prob = log_prob.mean()
        # log_prob = log_prob.sum() # 行動シーケンス全体を生成する確率なので，こちらが正しい．しかしlrを調整しなければ勾配爆発が起きる
        
        return log_prob
    
    def compute_action_distribution(self, obs: Dict, task: str) -> dist.Normal:
        """ガウス分布オブジェクトを返す"""
        batch = self._prepare_batch(obs, task)
        
        # SmolVLAから決定論的な平均行動を取得
        mean_action = self.smolvla_policy.select_action(batch)
        clamped_log_std = torch.clamp(self.log_std, min=np.log(self.min_std), max=None)
        std = torch.exp(clamped_log_std)
        if mean_action.dim() == 2:
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
        
        # オプティマイザー
        self.ppo_optimizer = torch.optim.AdamW([
            {'params': self.policy.smolvla_policy.parameters(), 'lr': config.get('smolvla_lr', 2.5e-6)},
            {'params': [self.policy.log_std], 'lr': config.get('log_std_lr', 1e-4)}
        ])
        self.value_optimizer = torch.optim.AdamW(
            self.value_network.parameters(), 
            lr=config['value_lr']
        )
        self.logger = logging.getLogger(__name__)
        self.epoch = 0
        self.video_frames = []
        
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
                    frame = self._render_frame(obs, reward, task_desc)
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
    
    def _render_frame(self, obs: Dict, reward, task_desc=None) -> Optional[np.ndarray]:
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
            # rewardを表示
            cv2.putText(combined, f"Reward: {reward:.2f}", (10, combined.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if task_desc:
                cv2.putText(combined, task_desc, (10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return combined
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
                    
                predicted_values_tensor = torch.stack(predicted_values).squeeze(-1)
                returns_tensor = torch.from_numpy(returns).float().to(self.device)
                # 価値関数のloss
                # print(f"predicted values: {predicted_values_tensor[0]}, returns: {returns_tensor[0]}")
                value_loss = F.mse_loss(predicted_values_tensor, returns_tensor)
            else:
                # 従来の状態価値関数の場合
                states = []
                for seq in sequences:
                    state = self._extract_state_tensor(seq.observation)
                    states.append(state)
                    
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
        return avg_value_loss
    
    def compute_flow_matching_loss(self, sequences: List[ActionSequence]) -> torch.Tensor:
        """SmolVLAのFlow Matching損失を計算"""
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        num_valid = 0
        
        for seq in sequences:
            # バッチを準備
            batch = self.policy._prepare_batch(seq.observation, seq.task)
            
            # アクションをテンソル形式に変換
            action_tensor = torch.from_numpy(seq.action).float().to(self.device).unsqueeze(0)
            loss, _ = self.policy.smolvla_policy.forward(
                {**batch, ACTION: action_tensor}
            )
            
            total_loss = total_loss + loss
            num_valid += 1
        
        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def update_policy(self, sequences: List[ActionSequence], advantages: np.ndarray) -> Dict:
        """PPO損失とFlow Matching損失を組み合わせた統合学習"""
        # アドバンテージの正規化
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        advantages_tensor = torch.from_numpy(advantages).float().to(self.device)
        
        # 古いlog probと古い分布を計算
        old_log_probs = []
        old_distributions = []
        for seq in sequences:
            log_prob = self.policy.compute_action_logprob(
                seq.observation, seq.action, seq.task
            )
            old_log_probs.append(log_prob.detach())
            old_dist = self.policy.compute_action_distribution(seq.observation, seq.task)
            # 分布のパラメータをdetachして新しい分布を作成
            detached_dist = dist.Normal(old_dist.loc.detach(), old_dist.scale.detach())
            old_distributions.append(detached_dist)
        
        old_log_probs_tensor = torch.stack(old_log_probs)
        
        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        total_kl_div = 0.0
        total_flow_matching_loss = 0.0
        ppo_epoch = 0
        
        if self.epoch > self.smolvla_warmup_epochs:
            for ppo_epoch in range(self.config.get('ppo_epochs', 4)):
                # 統合損失での更新
                ppo_loss, entropy_loss, kl_div, flow_matching_loss = self._update_unified(
                    sequences, advantages_tensor, old_log_probs_tensor, old_distributions
                )
                
                total_policy_loss += ppo_loss
                total_entropy_loss += entropy_loss
                total_kl_div += kl_div
                total_flow_matching_loss += flow_matching_loss
                # kl_divが大きすぎる場合は学習を中断
                if kl_div > self.config.get('kl_thresh', 100):
                    break
        
        num_epochs = ppo_epoch + 1
        avg_policy_loss = total_policy_loss / num_epochs
        avg_entropy_loss = total_entropy_loss / num_epochs
        avg_kl_div = total_kl_div / num_epochs
        avg_flow_matching_loss = total_flow_matching_loss / num_epochs
        
        return {
            'policy_loss': avg_policy_loss,
            'entropy_loss': avg_entropy_loss,
            'flow_matching_loss': avg_flow_matching_loss,
            'kl_divergence': avg_kl_div,
            'num_sequences': len(sequences),
            'avg_reward': np.mean([seq.reward for seq in sequences]),
            'avg_advantage': np.mean(advantages),
            'std_mean': torch.exp(self.policy.log_std).mean().item(),
        }
    
    def _update_unified(self, sequences: List[ActionSequence], advantages_tensor: torch.Tensor, old_log_probs_tensor: torch.Tensor, old_distributions: List) -> Tuple[float, float, float, float]:
        """PPO損失とFlow Matching損失を統合した更新"""
        # 新しい分布を作成
        new_distributions = []
        new_log_probs = []
        entropies = []
        
        for seq in sequences:
            # 新しい分布を取得
            new_dist = self.policy.compute_action_distribution(seq.observation, seq.task)
            new_distributions.append(new_dist)
            
            # log probを計算
            log_prob = self.policy.compute_action_logprob(
                seq.observation, seq.action, seq.task
            )
            new_log_probs.append(log_prob)
            
            # エントロピー計算（PyTorchの分布から）
            entropy = new_dist.entropy().sum()
            entropies.append(entropy)
        
        new_log_probs_tensor = torch.stack(new_log_probs)
        entropy_tensor = torch.stack(entropies)
        log_prob_diff = new_log_probs_tensor - old_log_probs_tensor
        
        # 重要度サンプリング比
        ratio = torch.exp(log_prob_diff)
        
        # KLダイバージェンスを計算
        kl_divs = []
        for old_dist, new_dist in zip(old_distributions, new_distributions):
            kl_div_sample = dist.kl_divergence(old_dist, new_dist).sum()
            kl_divs.append(kl_div_sample)
        
        kl_div = torch.stack(kl_divs).mean()
        # print(f"ratio min: {ratio.min().item()}, max: {ratio.max().item()}, mean: {ratio.mean().item()}")
        # print(f"advantages min: {advantages_tensor.min().item()}, max: {advantages_tensor.max().item()}, mean: {advantages_tensor.mean().item()}")
        # PPO loss
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(
            ratio, 
            1.0 - self.config['clip_epsilon'], 
            1.0 + self.config['clip_epsilon']
        ) * advantages_tensor
        
        policy_loss = -torch.min(surr1, surr2).mean() # これが巨大な値を取る．
        entropy_loss = -self.config.get('entropy_coef', 0.01) * entropy_tensor.mean()
        
        # Flow Matching損失の計算
        flow_matching_loss = self.compute_flow_matching_loss(sequences) * self.flow_matching_coef
        
        # 統合損失（PPO損失 + flow_matching_coef * Flow Matching損失）
        total_loss = policy_loss + entropy_loss + flow_matching_loss
        
        # 統合オプティマイザによる更新
        self.ppo_optimizer.zero_grad()
        total_loss.backward()
        
        if self.config.get('max_grad_norm', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.smolvla_policy.parameters()) + [self.policy.log_std], 
                self.config['max_grad_norm']
            )
        
        self.ppo_optimizer.step()
        
        return policy_loss.item(), entropy_loss.item(), kl_div.item(), flow_matching_loss.item()
    
    def train(self):
        self.logger.info(f"Starting PPO training for {self.config['epochs']} epochs")
        
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch}/{self.config['epochs']}")
            
            # データ収集
            sequences = self.collect_trajectories()
            # アドバンテージ計算
            advantages, returns = self.compute_advantages(sequences)
            # 価値関数の更新
            value_loss = self.update_value_function(sequences, returns)
            # 価値関数の更新
            policy_update_info = {'policy_loss': 0.0}
            policy_update_info = self.update_policy(sequences, advantages)
          
            # ログ記録
            log_dict = {
                'epoch': epoch,
                'value_loss': value_loss,
                'policy_loss': policy_update_info['policy_loss'],
                'avg_reward': np.mean([seq.reward for seq in sequences]),
                'num_sequences': len(sequences),
            }
            
            # 追加のメトリクスを含める
            if 'entropy_loss' in policy_update_info:
                log_dict.update({
                    'entropy_loss': policy_update_info['entropy_loss'],
                    'kl_divergence': policy_update_info['kl_divergence'],
                    'std_mean': policy_update_info['std_mean'],
                    'flow_matching_loss': policy_update_info.get('flow_matching_loss', 0.0),
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
    
    def save_checkpoint(self, epoch: int):
        checkpoint_dir = Path(self.config['checkpoint_dir']) / f"epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'policy_state_dict': self.policy.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'ppo_optimizer_state_dict': self.ppo_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': self.config
        }
        
        policy_path = checkpoint_dir / "checkpoint.pth"
        torch.save(checkpoint, policy_path)
        
        smolvla_path = checkpoint_dir / "smolvla_policy"
        self.policy.smolvla_policy.save_pretrained(smolvla_path)