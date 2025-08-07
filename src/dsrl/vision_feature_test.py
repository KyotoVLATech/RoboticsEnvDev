import numpy as np
import os
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os
import numpy as np
from PIL import Image
import torch
import wandb
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from env.genesis_env import GenesisEnv
from env.tasks.test import joints_name
from src.dsrl.feature_cnn import VisionFeatureTest1, VisionFeatureTest2

task_description = {
    "test": "Pick up a red cube and place it in a box.",
}

def expert_policy(env, stage):
    task = env._env
    if task.color == "red":
        cube_pos = task.cubeA.get_pos().cpu().numpy()
    elif task.color == "blue":
        cube_pos = task.cubeB.get_pos().cpu().numpy()
    elif task.color == "green":
        cube_pos = task.cubeC.get_pos().cpu().numpy()
    finder_pos = -0.02  # tighter grip
    quat = np.array([0, 1, 0, 0]) # Changed from [[0, 1, 0, 0]] to [0, 1, 0, 0]
    eef = task.eef
    if stage == "hover1":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.3])
        grip = np.array([0.04, 0.04])
    elif stage == "hover2":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.2])
        grip = np.array([0.04, 0.04])
    elif stage == "stabilize":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.1])
        grip = np.array([0.04, 0.04])
    elif stage == "grasp":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.1])  # lower slightly
        grip = np.array([finder_pos, finder_pos])  # close grip
    elif stage == "lift":
        target_pos = np.array([cube_pos[0], cube_pos[1], 0.25])
        grip = np.array([finder_pos, finder_pos])  # keep closed
    elif stage == "to_box":
        box_pos = task.box.get_pos().cpu().numpy()
        target_pos = box_pos + np.array([0.0, 0.0, 0.25])
        grip = np.array([finder_pos, finder_pos])
    elif stage == "stabilize_box":
        box_pos = task.box.get_pos().cpu().numpy()
        target_pos = box_pos + np.array([0.0, 0.0, 0.25])
        grip = np.array([finder_pos, finder_pos])
    elif stage == "release":
        box_pos = task.box.get_pos().cpu().numpy()
        target_pos = box_pos + np.array([0.0, 0.0, 0.25])
        grip = np.array([0.04, 0.04])
    else:
        raise ValueError(f"Unknown stage: {stage}")
    # Use IK to compute joint positions for the arm
    qpos = task.franka.inverse_kinematics(
        link=eef,
        pos=target_pos,
        quat=quat,
    ).cpu().numpy()
    qpos_arm = qpos[:-2]
    action = np.concatenate([qpos_arm, grip]) # Shape (9)
    return action.astype(np.float32)

def initialize_dataset(task, height, width):
    # Initialize dataset
    dict_idx = 0
    dataset_path = f"datasets/{task}_{dict_idx}"
    while os.path.exists(f"datasets/{task}_{dict_idx}"):
        dict_idx += 1
        dataset_path = f"datasets/{task}_{dict_idx}"
    lerobot_dataset = LeRobotDataset.create(
        repo_id=None,
        fps=30,
        root=dataset_path,
        robot_type="franka",
        use_videos=True,
        features={
            "observation.state": {"dtype": "float32", "shape": (9,), "names": joints_name},
            "action": {"dtype": "float32", "shape": (9,), "names": joints_name},
            "observation.images.front": {"dtype": "video", "shape": (height, width, 3), "names": ("height", "width", "channels")},
            "observation.images.side": {"dtype": "video", "shape": (height, width, 3), "names": ("height", "width", "channels")},
            "observation.images.eef": {"dtype": "video", "shape": (height, width, 3), "names": ("height", "width", "channels")},
        },
    )
    return lerobot_dataset

def get_task_info(task_desc: str) -> float:
    if 'green' in task_desc:
        return -1.0
    elif 'red' in task_desc:
        return 0.0
    elif 'blue' in task_desc:
        return 1.0
    else:
        print(f"Unknown task description: {task_desc}", file=sys.stderr)
        return 0.0

def main(task, stage_dict, observation_height=480, observation_width=640, episode_num=1, show_viewer=False):
    env = GenesisEnv(task=task, observation_height=observation_height, observation_width=observation_width, show_viewer=show_viewer)
    dataset = initialize_dataset(task, observation_height, observation_width)
    config = {
        "model_path": "smolvla_model.pth",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "smolvla_config_overrides": {}
    }
    test1 = VisionFeatureTest1(feature_dim=256).to(config['device'])
    test2 = VisionFeatureTest2(config).to(config['device'])
    opt1 = torch.optim.AdamW(test1.parameters(), lr=1e-4)
    opt2 = torch.optim.AdamW(test2.parameters(), lr=1e-4)
    ep = 0
    global_step = 0
    while ep < episode_num:
        env.reset()
        states, images_front, images_side, images_eef, actions = [], [], [], [], []
        save_flag = False
        episode_reward = 0.0
        for stage in stage_dict.keys():
            for t in range(stage_dict[stage]):
                action = expert_policy(env, stage)
                obs, reward, _, _, _ = env.step(action)
                obs['task_desk'] = env.get_task_description()
                obs['task_id'] = torch.tensor(get_task_info(obs['task_desk']), dtype=torch.float32)
                output1 = test1(obs)
                output2 = test2(obs)
                if env._env.color == "red":
                    target = env._env.cubeA.get_pos().cpu().numpy()
                elif env._env.color == "blue":
                    target = env._env.cubeB.get_pos().cpu().numpy()
                elif env._env.color == "green":
                    target = env._env.cubeC.get_pos().cpu().numpy()
                else:
                    raise ValueError(f"Unknown color: {env._env.color}")
                target -= env._env.eef.get_pos().cpu().numpy() # エンドエフェクタからの相対位置に変換
                loss1 = torch.nn.functional.mse_loss(output1, torch.tensor(target, dtype=torch.float32))
                loss2 = torch.nn.functional.mse_loss(output2, torch.tensor(target, dtype=torch.float32))
                opt1.zero_grad()
                loss1.backward()
                opt1.step()
                opt2.zero_grad()
                loss2.backward()
                opt2.step()
                wandb.log({
                    "loss1": loss1.item(),
                    "loss2": loss2.item(),
                    "global_step": global_step,
                })
                episode_reward += reward
                states.append(obs["agent_pos"])
                images_front.append(obs["observation.images.front"])
                images_side.append(obs["observation.images.side"])
                images_eef.append(obs["observation.images.eef"])
                actions.append(action)
                global_step += 1
                if reward >= 1.0:
                    save_flag = True
        if episode_reward >= 40.0:
            save_flag = True
        else:
            print(f"Episode reward: {episode_reward}, failed.")
            save_flag = False
        if not save_flag:
            continue
        ep += 1
        for i in range(len(states)):
            image_front = images_front[i]
            if isinstance(image_front, Image.Image):
                image_front = np.array(image_front)
            image_side = images_side[i]
            if isinstance(image_side, Image.Image):
                image_side = np.array(image_side)
            image_eef = images_eef[i]
            if isinstance(image_eef, Image.Image):
                image_eef = np.array(image_eef)
            dataset.add_frame(
                {
                    "observation.state": states[i].astype(np.float32),
                    "action": actions[i].astype(np.float32),
                    "observation.images.front": image_front,
                    "observation.images.side": image_side,
                    "observation.images.eef": image_eef,
                },
                task=env.get_task_description(),
            )
        dataset.save_episode()
    env.close()

if __name__ == "__main__":
    wandb.init(project="vision-feature-test")
    task = "simple_pick"
    stage_dict = {
        "hover1": 100,
        "hover2": 30,
        "stabilize": 40,
        "grasp": 20,
        "lift": 50,
    }
    main(task, stage_dict=stage_dict, observation_height=512, observation_width=512, episode_num=500, show_viewer=False)