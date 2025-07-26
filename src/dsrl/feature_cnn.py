import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class FeatureCNN(nn.Module):
    """
    画像特徴量抽出用の4層32ch CNN
    入力: (B, C, H, W)
    出力: (B, feature_dim)
    """
    def __init__(self, in_channels=3, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(32 * 8 * 16, feature_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class VisionNet(torch.nn.Module):
    """
    統合された特徴量ベクトルから画像を復元してCNN処理を行うネットワーク
    """
    def __init__(self, cnn, net, device):
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
        self.vlm_dim = 720
        self.proprioceptive_dim = 32  # SmolVLAの自己受容状態次元

    def forward(self, obs, state=None, info={}):
        obs_tensor = torch.from_numpy(obs).float().to(self.device).squeeze(0)
        # 統合ベクトルから各要素を分離: front_img + side_img + vlm_features + proprioceptive_state
        front_img_flat = obs_tensor[:, :self.front_img_size]
        side_img_flat = obs_tensor[:, self.front_img_size:self.front_img_size + self.side_img_size]
        vlm_features = obs_tensor[:, self.front_img_size + self.side_img_size:self.front_img_size + self.side_img_size + self.vlm_dim]
        proprioceptive_state = obs_tensor[:, self.front_img_size + self.side_img_size + self.vlm_dim:]
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
        # CNN特徴量抽出
        cnn_feat = self.cnn(concat_img)
        # 全ての特徴量を結合
        combined_features = torch.cat([cnn_feat, vlm_features, proprioceptive_state], dim=1)
        # MLPに通す
        output = self.net(combined_features)
        return output
