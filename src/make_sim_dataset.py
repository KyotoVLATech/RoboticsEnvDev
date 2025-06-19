import genesis as gs
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.genesis_env import GenesisEnv
from env.tasks.test import joints_name
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import os
import numpy as np
from PIL import Image

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
    if stage == "hover":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.2])  # hover safely
        grip = np.array([0.04, 0.04])  # open
    elif stage == "stabilize":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.1])
        grip = np.array([0.04, 0.04])  # still open
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
        },
    )
    return lerobot_dataset

def main(task, task_description, stage_dict, observation_height=480, observation_width=640, episode_num=1, show_viewer=False):
    env = None
    dataset = initialize_dataset(task, observation_height, observation_width)
    ep = 0
    while ep < episode_num:
        print(f"\n🎬 Starting episode {ep+1}")
        if ep % 10 == 0:
            # メモリリークを避けるために、10エピソードごとに環境をリセット
            if env is not None:
                env.close()
            env = GenesisEnv(task=task, observation_height=observation_height, observation_width=observation_width, show_viewer=show_viewer)
        env.reset()
        states, images_front, images_side, actions = [], [], [], []
        save_flag = False
        episode_reward = 0.0
        for stage in stage_dict.keys():
            print(f"  Stage: {stage}")
            for t in range(stage_dict[stage]):
                action = expert_policy(env, stage)
                obs, reward, _, _, _ = env.step(action)
                episode_reward += reward
                states.append(obs["agent_pos"])
                images_front.append(obs["observation.images.front"])
                images_side.append(obs["observation.images.side"])
                actions.append(action)
                if reward >= 1.0:
                    save_flag = True
        if task == "simple_pick":
            if episode_reward >= 80.0:
                print(f"Episode reward: {episode_reward}, success!")
                save_flag = True
            else:
                print(f"Episode reward: {episode_reward}, failed.")
                save_flag = False
        # デバッグ用
        env.save_video(file_name=f"video", fps=30)

        if not save_flag:
            print(f"🚫 Skipping episode {ep+1}")
            continue
        print(f"✅ Saving episode {ep+1}")
        ep += 1

        for i in range(len(states)):
            image_front = images_front[i]
            if isinstance(image_front, Image.Image):
                image_front = np.array(image_front)
            image_side = images_side[i]
            if isinstance(image_side, Image.Image):
                image_side = np.array(image_side)
            task_desc = None
            if task in task_description:
                task_desc = task_description[task]
            elif task == "simple_pick":
                task_desc = f"Pick up a {env._env.color} cube."
            dataset.add_frame(
                {
                    "observation.state": states[i].astype(np.float32),
                    "action": actions[i].astype(np.float32),
                    "observation.images.front": image_front,
                    "observation.images.side": image_side,
                },
                task=task_desc,
            )
        dataset.save_episode()
    env.close()

if __name__ == "__main__":
    task = "simple_pick" # [test, simple_pick]
    state_dict = None
    if task == "test":
        stage_dict = { # 350
            "hover": 100, # cubeの上に手を持っていく
            "stabilize": 40, # cubeの上で手を安定させる
            "grasp": 20, # cubeを掴む
            "lift": 50, # cubeを持ち上げる
            "to_box": 60, # cubeを箱の上に持っていく
            "stabilize_box": 20, # cubeを箱の上で安定させる
            "release": 60, # cubeを離す
        }
    elif task == "simple_pick":
        stage_dict = { # 210
            "hover": 100, # cubeの上に手を持っていく
            "stabilize": 40, # cubeの上で手を安定させる
            "grasp": 20, # cubeを掴む
            "lift": 50, # cubeを持ち上げる
        }
    main(task, task_description, stage_dict=stage_dict, observation_height=512, observation_width=512, episode_num=100, show_viewer=False)