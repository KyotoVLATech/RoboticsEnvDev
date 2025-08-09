import numpy as np
import os
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os
import numpy as np
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.genesis_env import GenesisEnv
from env.tasks.test import joints_name

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
    dataset_path = f"datasets/vision_test_{dict_idx}"
    while os.path.exists(f"datasets/vision_test_{dict_idx}"):
        dict_idx += 1
        dataset_path = f"datasets/vision_test_{dict_idx}"
    lerobot_dataset = LeRobotDataset.create(
        repo_id=None,
        fps=30,
        root=dataset_path,
        robot_type="franka",
        use_videos=True,
        features={
            "target": {"dtype": "float", "shape": (3,), "names": ("x", "y", "z")},
            "observation.state": {"dtype": "float32", "shape": (9,), "names": joints_name},
            "action": {"dtype": "float32", "shape": (9,), "names": joints_name},
            "observation.images.front": {"dtype": "video", "shape": (height, width, 3), "names": ("height", "width", "channels")},
            "observation.images.side": {"dtype": "video", "shape": (height, width, 3), "names": ("height", "width", "channels")},
            "observation.images.eef": {"dtype": "video", "shape": (height, width, 3), "names": ("height", "width", "channels")},
        },
    )
    return lerobot_dataset

def main(task, stage_dict, observation_height=480, observation_width=640, episode_num=1, show_viewer=False):
    env = GenesisEnv(task=task, observation_height=observation_height, observation_width=observation_width, show_viewer=show_viewer)
    dataset = initialize_dataset(task, observation_height, observation_width)
    ep = 0
    while ep < episode_num:
        print(f"\n🎬 Starting episode {ep+1}")
        env.reset()
        if env._env.color == "red":
            target = env._env.cubeA.get_pos()
        elif env._env.color == "blue":
            target = env._env.cubeB.get_pos()
        elif env._env.color == "green":
            target = env._env.cubeC.get_pos()
        else:
            raise ValueError(f"Unknown color: {env._env.color}")
        images_front, images_side, images_eef, targets, states, actions = [], [], [], [], [], []
        save_flag = False
        episode_reward = 0.0
        for stage in stage_dict.keys():
            for t in range(stage_dict[stage]):
                action = expert_policy(env, stage)
                obs, reward, _, _, _ = env.step(action)
                episode_reward += reward
                states.append(obs["agent_pos"])
                images_front.append(obs["observation.images.front"])
                images_side.append(obs["observation.images.side"])
                images_eef.append(obs["observation.images.eef"])
                actions.append(action)
                targets.append(target.cpu().numpy())
                if reward >= 1.0:
                    save_flag = True
        if episode_reward >= 40.0:
            save_flag = True
        else:
            save_flag = False
        if not save_flag:
            continue
        ep += 1

        for i in range(len(images_front)):
            image_front = images_front[i]
            if isinstance(image_front, Image.Image):
                image_front = np.array(image_front)
            image_side = images_side[i]
            if isinstance(image_side, Image.Image):
                image_side = np.array(image_side)
            image_eef = images_eef[i]
            if isinstance(image_eef, Image.Image):
                image_eef = np.array(image_eef)
            target = targets[i].astype(np.float64)
            dataset.add_frame(
                {
                    "target": target,
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
    ep_num = 100
    stage_dict = {
        "hover1": 100, # cubeの上に手を持っていく
        "hover2": 30, # cubeの上で手を安定させる
        "stabilize": 40, # cubeの上で手を安定させる
        "grasp": 20, # cubeを掴む
        "lift": 50, # cubeを持ち上げる
    }
    main("simple_pick", stage_dict=stage_dict, observation_height=512, observation_width=512, episode_num=ep_num, show_viewer=False)