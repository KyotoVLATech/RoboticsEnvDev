import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class ResidualBlock(nn.Module):
    """
    残差ブロック
    """
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.act1 = TeLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups, channels)
        self.act2 = TeLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups, channels)
        self.act3 = TeLU()

    def forward(self, x):
        residual = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x += residual # 残差接続
        x = self.act3(x)
        return x

class FeatureCNN(nn.Module):
    """
    入力: (B, C, H, W)
    出力: (B, feature_dim)
    """
    def __init__(self, in_channels=3, feature_dim=256, num_groups=32, use_residual=True):
        super().__init__()
        self.feature_dim = feature_dim
        # 畳み込み層の定義
        self.conv1 = self._make_downsample_block(in_channels, 64, num_groups=num_groups) # H, W -> H/2, W/2
        self.conv2 = self._make_downsample_block(64, 128, num_groups=num_groups)         # -> H/4, W/4
        self.conv3 = self._make_downsample_block(128, 256, num_groups=num_groups)        # -> H/8, W/8
        self.conv4 = self._make_downsample_block(256, 256, num_groups=num_groups)        # -> H/16, W/16
        self.use_residual = use_residual
        # 残差ブロック
        if use_residual:
            self.resblock1 = ResidualBlock(256, num_groups=num_groups)
            self.resblock2 = ResidualBlock(256, num_groups=num_groups)

        # 空間次元を要約し、最終的な特徴ベクトルに変換
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, feature_dim)

    def _make_downsample_block(self, in_channels, out_channels, num_groups=32):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            TeLU()
        )

    def forward(self, x):
        # 畳み込みでダウンサンプリング
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # 残差ブロックで特徴を洗練
        if self.use_residual:
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
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
        elif isinstance(obs, torch.Tensor):
            obs_tensor = obs.float().to(self.device)
        # 統合ベクトルから各要素を分離: front_img + side_img + vlm_features + proprioceptive_state
        front_img_flat = obs_tensor[:, :self.front_img_size]
        side_img_flat = obs_tensor[:, self.front_img_size:self.front_img_size + self.side_img_size]
        vlm_features = obs_tensor[:, self.front_img_size + self.side_img_size:self.front_img_size + self.side_img_size + self.vlm_dim]
        proprioceptive_state = obs_tensor[:, self.front_img_size + self.side_img_size + self.vlm_dim:-1]
        task_id = obs_tensor[:, -1]  # 最後の要素はタスクID
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
        # proprioceptive_stateを正規化
        proprioceptive_state /= np.pi
        # vlm_featuresを正規化
        vlm_features /= 5.0
        # 全ての特徴量を結合
        combined_features = torch.cat([cnn_feat, vlm_features, proprioceptive_state, task_id.unsqueeze(1)], dim=1)  # (B, feature_dim + vlm_dim + proprio_dim + 1)
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

class VisionFeatureTest1(nn.Module):
    """
    CNNを用いて画像から特徴量を抽出するテスト用ネットワーク
    """
    def __init__(self, config):
        super().__init__()
        feature_dim = config["test1_feature_dim"]
        self.device = config["device"]
        self.config = config
        self.cnn = FeatureCNN(in_channels=3, feature_dim=feature_dim, use_residual=config["use_residual"])
        task_dim = 1 if config["use_task_id"] else 720
        if config["use_mlp1"]:
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim + task_dim, 512),
                TeLU(),
                nn.Linear(512, 256),
                TeLU(),
                nn.Linear(256, 128),
                TeLU(),
            )
            self.fc = nn.Linear(128, 3)
        else:
            self.fc = nn.Linear(feature_dim + task_dim, 3)

    def forward(self, input_dict):
        images = []
        for key in ["observation.images.front", "observation.images.side", "observation.images.eef"]:
            image = input_dict[key]
            image = image.unsqueeze(0)  # (1, C, H, W)
            image = torch.nn.functional.interpolate(image, size=(128, 128), mode='bilinear', align_corners=False)
            images.append(image)
        concat_img = torch.cat(images, dim=2).to(self.device)
        features = self.cnn(concat_img)
        if self.config["use_task_id"]:
            task_feature = input_dict["task_index"].unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            task_feature = input_dict["vlm_features"]
        features = torch.cat([features, task_feature], dim=1)
        if self.config["use_mlp1"]:
            features = self.mlp(features)
        output = self.fc(features)
        return output

class VisionFeatureTest2(nn.Module):
    """
    SmolVLAのイメージトークナイザを利用して画像特徴量を抽出するテスト用ネットワーク
    """
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.config = config
        task_dim = 1 if config["use_task_id"] else 720
        if config["use_mlp2"]:
            self.mlp = nn.Sequential(
                nn.Linear(960*3 + task_dim, 512),
                TeLU(),
                nn.Linear(512, 256),
                TeLU(),
                nn.Linear(256, 128),
                TeLU(),
            )
            self.fc = nn.Linear(128, 3)
        else:
            self.fc = nn.Linear(960*3 + task_dim, 3)

    def forward(self, input_dict):
        if self.config["use_task_id"]:
            task_feature = input_dict["task_index"].unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            task_feature = input_dict["vlm_features"]
        features = input_dict["vit_features"]
        features = torch.cat([features, task_feature], dim=1)
        if self.config["use_mlp2"]:
            features = self.mlp(features)
        output = self.fc(features)
        return output

# 活性化関数
class TeLU(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))

class ReCA(nn.Module):
    def __init__(self, a=0.2, b=0.3):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))

    def forward(self, x):
        # 論文の式に基づく近似例（a, b を学習）
        return torch.max(x, self.a * x + self.b * torch.relu(x))

class S4(nn.Module):
    def __init__(self, k=1.0):
        super().__init__()
        self.k = k

    def forward(self, x):
        # 負の領域: sigmoid, 正の領域: softsign
        return torch.where(x < 0, torch.sigmoid(self.k * x), x / (1 + torch.abs(x)))

class GCU(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x * torch.cos(self.alpha * x)