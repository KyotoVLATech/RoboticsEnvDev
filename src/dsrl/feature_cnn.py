import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2D(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2D(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x += residual

class FeatureCNN(nn.Module):
    """
    入力: (B, C, H, W)
    出力: (B, feature_dim)
    """
    def __init__(self, in_channels=3, feature_dim=512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.Flatten(),
        )
        self.fc = nn.Linear(256, feature_dim)

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.cnn(x)
        print("CNN output shape:", x.shape)  # デバッグ用
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
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
        elif isinstance(obs, torch.Tensor):
            obs_tensor = obs.float().to(self.device)
        # 統合ベクトルから各要素を分離: front_img + side_img + vlm_features + proprioceptive_state
        front_img_flat = obs_tensor[:, :self.front_img_size]
        side_img_flat = obs_tensor[:, self.front_img_size:self.front_img_size + self.side_img_size]
        vlm_features = obs_tensor[:, self.front_img_size + self.side_img_size:self.front_img_size + self.side_img_size + self.vlm_dim]
        proprioceptive_state = obs_tensor[:, self.front_img_size + self.side_img_size + self.vlm_dim:]
        print("vlm features max:", vlm_features.max(), "min:", vlm_features.min()) # max 4.4 , min -4.5
        print("proprioceptive state max:", proprioceptive_state.max(), "min:", proprioceptive_state.min()) # max 3.14, min, -0.35
        # 正規化処理を入れた方が良い
        
        # 画像を元の形状に復元
        front_img = front_img_flat.view(-1, self.img_channels, self.img_height, self.img_width)
        side_img = side_img_flat.view(-1, self.img_channels, self.img_height, self.img_width)
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
        # 全ての特徴量を結合
        combined_features = torch.cat([cnn_feat, vlm_features, proprioceptive_state], dim=1)
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