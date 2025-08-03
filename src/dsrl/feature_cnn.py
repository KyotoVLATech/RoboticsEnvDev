import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    残差ブロック
    """
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups, channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual # 残差接続
        x = F.relu(x)
        return x

class FeatureCNN(nn.Module):
    """
    入力: (B, C, H, W)
    出力: (B, feature_dim)
    """
    def __init__(self, in_channels=3, feature_dim=256, num_groups=32):
        super().__init__()
        self.feature_dim = feature_dim
        # 畳み込み層の定義
        self.conv1 = self._make_downsample_block(in_channels, 64, num_groups=num_groups) # H, W -> H/2, W/2
        self.conv2 = self._make_downsample_block(64, 128, num_groups=num_groups)         # -> H/4, W/4
        self.conv3 = self._make_downsample_block(128, 256, num_groups=num_groups)        # -> H/8, W/8
        self.conv4 = self._make_downsample_block(256, 256, num_groups=num_groups)        # -> H/16, W/16

        # 残差ブロック
        self.resblock1 = ResidualBlock(256, num_groups=num_groups)
        self.resblock2 = ResidualBlock(256, num_groups=num_groups)

        # 空間次元を要約し、最終的な特徴ベクトルに変換
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, feature_dim)

    def _make_downsample_block(self, in_channels, out_channels, num_groups=32):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 畳み込みでダウンサンプリング
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # 残差ブロックで特徴を洗練
        x = self.resblock1(x)
        x = self.resblock2(x)
        
        # Global Average Poolingで空間情報を要約
        x = self.global_avg_pool(x)
        
        # フラット化して全結合層に入力
        x = x.view(x.size(0), -1) # (B, 256, 1, 1) -> (B, 256)
        x = self.fc(x)
        
        return x

class VisionNet(torch.nn.Module):
    """
    統合された特徴量ベクトルから画像を復元してCNN処理を行うネットワーク
    """
    def __init__(self, cnn, net, vlm_dim, device):
        super().__init__()
        self.cnn = cnn
        self.net = net
        self.output_dim = net.output_dim  # MLPの出力次元
        self.device = device
        
        # 画像サイズの定義（SmolVLAWrapperと一致させる）
        self.img_height = 512
        self.img_width = 512
        self.img_channels = 3
        self.front_img_size = self.img_height * self.img_width * self.img_channels
        self.side_img_size = self.img_height * self.img_width * self.img_channels
        self.vlm_dim = vlm_dim

    def forward(self, obs, state=None, info={}):
        # 必要なキーが存在するかチェック
        required_keys = ["front_img", "side_img", "vlm_features", "proprioceptive_state", "task_info"]
        if all(hasattr(obs, key) or (hasattr(obs, 'keys') and key in obs.keys()) for key in required_keys):
            front_img = obs["front_img"].float().to(self.device)  # (B, C, H, W)
            side_img = obs["side_img"].float().to(self.device)    # (B, C, H, W)
            vlm_features = obs["vlm_features"].float().to(self.device).squeeze(0) if len(obs["vlm_features"].shape) == 3 else obs["vlm_features"].float().to(self.device)  # (B, vlm_dim)
            proprioceptive_state = obs["proprioceptive_state"].float().to(self.device).squeeze(0) if len(obs["proprioceptive_state"].shape) == 3 else obs["proprioceptive_state"].float().to(self.device)  # (B, proprio_dim)
            task_id = obs["task_info"].float().to(self.device).squeeze(0) if len(obs["task_info"].shape) == 3 else obs["task_info"].float().to(self.device)  # (B, 1)
        else:
            raise ValueError("obs must be a dict with keys: front_img, side_img, vlm_features, proprioceptive_state, task_info")
        # 画像をリサイズしてCNN用に準備
        front_img_resized = torch.nn.functional.interpolate(front_img, size=(128, 128), mode='bilinear', align_corners=False)
        side_img_resized = torch.nn.functional.interpolate(side_img, size=(128, 128), mode='bilinear', align_corners=False)
        # 横方向に結合 (128, 256, 3)
        concat_img = torch.cat([front_img_resized, side_img_resized], dim=3)  # (B, C, H, W*2)
        # 0-1の範囲を-1~1に正規化
        concat_img = concat_img * 2 - 1
        self.save_img(concat_img)
        # CNN特徴量抽出
        cnn_feat = self.cnn(concat_img)
        # proprioceptive_stateを正規化
        proprioceptive_state = proprioceptive_state / np.pi
        # vlm_featuresを正規化
        vlm_features = vlm_features / 5.0
        # 全ての特徴量を結合
        combined_features = torch.cat([cnn_feat, vlm_features, proprioceptive_state, task_id], dim=1)  # (B, feature_dim + vlm_dim + proprio_dim + 1)
        # MLPに通す
        output = self.net(combined_features)
        return output

    def save_img(self, img_tensor):
        """
        デバッグ用に画像を保存する関数
        """
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.cpu().numpy() # Tensor -> NumPy変換
            img_np = ((img_np + 1) / 2 * 255).astype(np.uint8) # -1~1 float -> 0~255 uint8変換
            img_np = img_np[0] # 1枚目の画像を取得
            img_np = np.transpose(img_np, (1, 2, 0)) # チャンネルを最後に移動
            img_pil = Image.fromarray(img_np)
            img_pil.save("debug_image.png")
