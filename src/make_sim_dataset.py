# python3 -m src.make_sim_dataset

import os
from typing import Any

import genesis as gs
import numpy as np
from PIL import Image

from env.genesis_env import GenesisEnv
from env.tasks.sound import joints_name
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def expert_policy(env: GenesisEnv, stage: str) -> np.ndarray:
    task: Any = env._env  # Assuming env._env can be of various types, use Any
    cube_pos: np.ndarray = task.cubeA.get_pos().cpu().numpy()
    box_pos: np.ndarray = task.box.get_pos().cpu().numpy()
    # motors_dof = task.motors_dof
    # fingers_dof = task.fingers_dof
    finder_pos: float = -0.02  # tighter grip
    quat: np.ndarray = np.array(
        [0, 1, 0, 0]
    )  # Changed from [[0, 1, 0, 0]] to [0, 1, 0, 0]
    eef: Any = task.eef  # Assuming task.eef can be of various types, use Any

    target_pos: np.ndarray
    grip: np.ndarray
    # === Stage definitions ===
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
        target_pos = box_pos + np.array([0.0, 0.0, 0.25])
        grip = np.array([finder_pos, finder_pos])
    elif stage == "stabilize_box":
        target_pos = box_pos + np.array([0.0, 0.0, 0.25])
        grip = np.array([finder_pos, finder_pos])
    elif stage == "release":
        target_pos = box_pos + np.array([0.0, 0.0, 0.25])
        grip = np.array([0.04, 0.04])
    else:
        raise ValueError(f"Unknown stage: {stage}")
    # Use IK to compute joint positions for the arm
    qpos: np.ndarray = (
        task.franka.inverse_kinematics(
            link=eef,
            pos=target_pos,
            quat=quat,
        )
        .cpu()
        .numpy()
    )
    qpos_arm: np.ndarray = qpos[:-2]
    action: np.ndarray = np.concatenate([qpos_arm, grip])  # Shape (9)
    return action.astype(np.float32)


def initialize_dataset(task: str, height: int, width: int) -> LeRobotDataset:
    # Initialize dataset
    dict_idx: int = 0
    dataset_path: str = f"datasets/{task}_{dict_idx}"
    while os.path.exists(f"datasets/{task}_{dict_idx}"):
        dict_idx += 1
        dataset_path = f"datasets/{task}_{dict_idx}"
    lerobot_dataset: LeRobotDataset = LeRobotDataset.create(
        repo_id=None,
        fps=30,
        root=dataset_path,
        robot_type="franka",
        use_videos=True,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (9,),
                "names": joints_name,
            },
            "action": {"dtype": "float32", "shape": (9,), "names": joints_name},
            "observation.images.front": {
                "dtype": "video",
                "shape": (height, width, 3),
                "names": ("height", "width", "channels"),
            },
            "observation.images.side": {
                "dtype": "video",
                "shape": (height, width, 3),
                "names": ("height", "width", "channels"),
            },
            "observation.images.sound": {
                "dtype": "video",
                "shape": (height, width, 3),
                "names": ("height", "width", "channels"),
            },
        },
    )
    return lerobot_dataset


def main(
    task: str,
    stage_dict: dict[str, int],
    observation_height: int = 480,
    observation_width: int = 640,
    episode_num: int = 1,
    show_viewer: bool = False,
) -> None:
    gs.init(backend=gs.gpu, precision="32")  # cpuの方が早い？
    env: GenesisEnv | None = None
    dataset: LeRobotDataset = initialize_dataset(
        task, observation_height, observation_width
    )
    dummy_dataset: LeRobotDataset | None = None
    if task == "sound":
        dummy_dataset = initialize_dataset(
            "dummy", observation_height, observation_width
        )
    ep: int = 0
    while ep < episode_num:
        print(f"\n🎬 Starting episode {ep+1}")
        if ep % 10 == 0:
            # メモリリークを避けるために、10エピソードごとに環境をリセット
            if env is not None:
                env.close()
            env = GenesisEnv(
                task=task,
                observation_height=observation_height,
                observation_width=observation_width,
                show_viewer=show_viewer,
            )
        env.reset()
        states: list[np.ndarray] = []
        images_front: list[Any] = []  # Can be PIL.Image or np.ndarray
        images_side: list[Any] = []  # Can be PIL.Image or np.ndarray
        images_sound: list[Any] = []  # Can be PIL.Image or np.ndarray
        actions: list[np.ndarray] = []
        reward_greater_than_zero: bool = False
        for stage in stage_dict.keys():
            print(f"  Stage: {stage}")
            for t in range(stage_dict[stage]):
                action: np.ndarray = expert_policy(env, stage)
                obs: dict[str, Any]
                reward: float
                obs, reward, _, _, _ = env.step(action)
                states.append(obs["agent_pos"])
                images_front.append(obs["front"])
                images_side.append(obs["side"])
                images_sound.append(obs["sound"])
                actions.append(action)
                if reward > 0:
                    reward_greater_than_zero = True
        # デバッグ用
        # env.save_video(file_name=f"video", fps=30)

        if not reward_greater_than_zero:
            print(f"🚫 Skipping episode {ep+1} — reward was always 0")
            continue
        print(f"✅ Saving episode {ep+1} — reward > 0 observed")
        ep += 1

        for i in range(len(states)):
            image_front_item: Any = images_front[i]
            image_front_np: np.ndarray
            if isinstance(image_front_item, Image.Image):
                image_front_np = np.array(image_front_item)
            else:
                image_front_np = image_front_item

            image_side_item: Any = images_side[i]
            image_side_np: np.ndarray
            if isinstance(image_side_item, Image.Image):
                image_side_np = np.array(image_side_item)
            else:
                image_side_np = image_side_item

            image_sound_item: Any = images_sound[i]
            image_sound_np: np.ndarray
            if isinstance(image_sound_item, Image.Image):
                image_sound_np = np.array(image_sound_item)
            else:
                image_sound_np = image_sound_item

            dataset.add_frame(
                {
                    "observation.state": states[i].astype(np.float32),
                    "action": actions[i].astype(np.float32),
                    "observation.images.front": image_front_np,
                    "observation.images.side": image_side_np,
                    "observation.images.sound": image_sound_np,
                    "task": "pick cube with sound",
                }
            )
            if task == "sound" and dummy_dataset is not None:
                dummy_dataset.add_frame(
                    {
                        "observation.state": states[i].astype(np.float32),
                        "action": actions[i].astype(np.float32),
                        "observation.images.front": image_front_np,
                        "observation.images.side": image_side_np,
                        "observation.images.sound": np.zeros_like(image_sound_np),
                        "task": "pick cube without sound",
                    }
                )
        dataset.save_episode()
        if task == "sound" and dummy_dataset is not None:
            dummy_dataset.save_episode()
    if env is not None:
        env.close()


if __name__ == "__main__":
    task = "test"  # "test" or "sound"
    # 20秒くらいのタスクを想定 → 合計600フレーム
    stage_dict = {
        "hover": 100,  # cubeの上に手を持っていく
        "stabilize": 60,  # cubeの上で手を安定させる
        "grasp": 20,  # cubeを掴む
        "lift": 60,  # cubeを持ち上げる
        "to_box": 100,  # cubeを箱の上に持っていく
        "stabilize_box": 60,  # cubeを箱の上で安定させる
        "release": 60,  # cubeを離す
    }
    main(
        task=task,
        stage_dict=stage_dict,
        observation_height=480,
        observation_width=640,
        episode_num=1,
        show_viewer=False,
    )
