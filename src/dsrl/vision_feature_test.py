import numpy as np
import os
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os
import numpy as np
import torch
import wandb
import tqdm
from safetensors import safe_open
from safetensors.torch import save_file
from grams import Grams
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.dsrl.feature_cnn import VisionFeatureTest1, VisionFeatureTest2
from src.dsrl.custom_policy import SmolVLAWrapper, load_smolvla_model

def main(config):
    if config["use_wandb"]:
        wandb.init(
            project="vision-feature-test",
            config=config
        )
    dataset = LeRobotDataset(
        repo_id=config['data_path'],
        root=config['data_path']
    )
    smolvla_policy = load_smolvla_model(
        config['model_path'],
    )
    smolvla_wrapper = SmolVLAWrapper(smolvla_policy, config['device'])
    test1 = VisionFeatureTest1(config).to(config['device'])
    test2 = VisionFeatureTest2(config).to(config['device'])
    # opt1 = torch.optim.RAdam(test1.parameters(), lr=config["lr"])
    # opt2 = torch.optim.RAdam(test2.parameters(), lr=config["lr"])
    opt1 = Grams(test1.parameters(), lr=config["lr"])
    opt2 = Grams(test2.parameters(), lr=config["lr"])
    print("Dataset size: ", len(dataset))
    # Prepare VLA feature
    vla_features_path = "src/dsrl/vla_features.safetensors"
    if not os.path.exists(vla_features_path):
        vit_features = []
        vlm_features = []
        for i in tqdm.tqdm(range(len(dataset)), desc="Extracting VLA features"):
            data = dataset[i]
            data = smolvla_wrapper._prepare_batch(data, data["task"])
            with torch.no_grad():
                vit_feat = smolvla_wrapper._extract_vision_features(data)
                vlm_feat = smolvla_wrapper._extract_vlm_features(data)
                vit_features.append(vit_feat.cpu())
                vlm_features.append(vlm_feat.cpu())
        # safetensorsで保存するためにテンソルを結合
        vit_tensor = torch.stack(vit_features, dim=0)  # (num_samples, feature_dim)
        vlm_tensor = torch.stack(vlm_features, dim=0)  # (num_samples, feature_dim)
        tensors = {
            "vit_features": vit_tensor,
            "vlm_features": vlm_tensor
        }
        save_file(tensors, vla_features_path)
    else:
        # safetensorsから読み込み
        with safe_open(vla_features_path, framework="pt", device="cpu") as f:
            vit_features = f.get_tensor("vit_features")
            vlm_features = f.get_tensor("vlm_features")
    random_indices = np.random.permutation(len(dataset)).tolist()
    test_indices = random_indices[:config["test_data_num"]]
    learning_indices = random_indices[config["test_data_num"]:]
    global_step = 0
    for i in range(config['epoch']):
        print(f"Epoch {i+1}/{config['epoch']}")
        for idx in tqdm.tqdm(learning_indices):
            data = dataset[idx]
            data["vit_features"] = vit_features[idx].to(config["device"])
            data["vlm_features"] = vlm_features[idx].to(config["device"])
            output1 = test1(data).squeeze()
            output2 = test2(data).squeeze()
            target = data["target"].to(config["device"])
            loss1 = torch.nn.functional.mse_loss(output1, target)
            loss2 = torch.nn.functional.mse_loss(output2, target)
            opt1.zero_grad()
            loss1.backward()
            opt1.step()
            opt2.zero_grad()
            loss2.backward()
            opt2.step()
            if config["use_wandb"]:
                wandb.log({
                    "train/loss1": loss1.item(),
                    "train/loss2": loss2.item(),
                    "global_step": global_step,
                })
            if global_step % config['test_interval'] == 0:
                test1.eval()
                test2.eval()
                with torch.no_grad():
                    dist_list1 = []
                    dist_list2 = []
                    for test_idx in test_indices:
                        test_data = dataset[test_idx]
                        test_data["vit_features"] = vit_features[test_idx].to(config["device"])
                        test_data["vlm_features"] = vlm_features[test_idx].to(config["device"])
                        output1 = test1(test_data).squeeze()
                        output2 = test2(test_data).squeeze()
                        target = test_data["target"].to(config["device"])
                        # 距離を計算
                        distance1 = torch.norm(output1 - target)
                        distance2 = torch.norm(output2 - target)
                        dist_list1.append(distance1.item())
                        dist_list2.append(distance2.item())
                    if config["use_wandb"]:
                        wandb.log({
                            "test/dist1_mean": np.mean(dist_list1),
                            "test/dist1_std": np.std(dist_list1),
                            "test/dist2_mean": np.mean(dist_list2),
                            "test/dist2_std": np.std(dist_list2),
                            "global_step": global_step,
                        })
                test1.train()
                test2.train()
            global_step += 1

if __name__ == "__main__":
    config = {
        "use_wandb": True,
        "epoch": 2,
        "test_interval": 1000, # steps
        "test_data_num": 100,
        "data_path": "datasets/vision_test_0",
        "model_path": "outputs/train/smolvla_simple_pick_0/checkpoints/last/pretrained_model",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-4,
        "test1_feature_dim": 256,
        "use_residual": False, # Test1にResidualを使用するかどうか
        "use_task_id": False, # タスクIDを使用するかどうか
        "use_mlp1": True, # Test1にMLPを使用するかどうか
        "use_mlp2": True, # Test2にMLPを使用するかどうか
    }
    main(config)
