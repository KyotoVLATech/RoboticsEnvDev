import numpy as np
from PIL import Image
import os
import copy
from roboenv.env.genesis_env import GenesisEnv
from roboenv.env.tasks.normal import joints_name, AGENT_DIM
from lerobot.datasets.lerobot_dataset import LeRobotDataset

saved_cube_pos = None
is_first_call = True

def build_balanced_episode_configs(task, episode_num):
    variants = None
    if "normal" in task and "fix" not in task:
        variants = [
            {"reset_options": {"color": color}}
            for color in ["red", "green", "blue"]
        ]

    if not variants:
        return [{"reset_options": None} for _ in range(episode_num)]

    episode_configs = []
    full_cycles, remainder = divmod(episode_num, len(variants))

    for _ in range(full_cycles):
        cycle_variants = [copy.deepcopy(variant) for variant in variants]
        np.random.shuffle(cycle_variants)
        episode_configs.extend(cycle_variants)

    if remainder:
        cycle_variants = [copy.deepcopy(variant) for variant in variants]
        np.random.shuffle(cycle_variants)
        episode_configs.extend(cycle_variants[:remainder])

    return episode_configs

def get_left_and_right_cube_names(task):
    cube_positions = {
        "cubeR": task.cubeR.get_pos().cpu().numpy(),
        "cubeG": task.cubeG.get_pos().cpu().numpy(),
    }
    ordered_names = sorted(cube_positions, key=lambda name: cube_positions[name][1], reverse=True)
    return ordered_names[0], ordered_names[1]

def expert_policy(env, stage, target_cube_name=None):
    global saved_cube_pos, is_first_call
    task = env._env
    
    # ターゲットのCubeとBoxを決定
    target_cube_pos = None
    target_box_pos = None
    
    if target_cube_name is not None:
        if target_cube_name == "cubeR":
            target_cube_pos = task.cubeR.get_pos().cpu().numpy()
        elif target_cube_name == "cubeG":
            target_cube_pos = task.cubeG.get_pos().cpu().numpy()
        elif target_cube_name == "cubeB":
            target_cube_pos = task.cubeB.get_pos().cpu().numpy()
        target_box_pos = task.box.get_pos().cpu().numpy()

    # NormalTask or fallback
    if target_cube_pos is None:
        if task.color == "red":
            target_cube_pos = task.cubeR.get_pos().cpu().numpy()
        elif task.color == "blue":
            target_cube_pos = task.cubeB.get_pos().cpu().numpy()
        elif task.color == "green":
            target_cube_pos = task.cubeG.get_pos().cpu().numpy()
        target_box_pos = task.box.get_pos().cpu().numpy()
        
    cube_pos = target_cube_pos
    box_pos = target_box_pos
    grip_close = np.array([-0.02, -0.02])  # tighter grip for Franka
    grip_open = np.array([0.04, 0.04])
    quat = np.array([0, 1, 0, 0], dtype=np.float32)  # Franka gripper orientation
    quat /= np.linalg.norm(quat)
    eef = task.eef
    # === Stage definitions ===
    if stage == "hover":
        is_first_call = True
        target_pos = cube_pos + np.array([0.0, 0.0, 0.2])  # hover safely
        grip = grip_open
    elif stage == "stabilize":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.1])
        grip = grip_open  # still open
    elif stage == "grasp":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.1])  # lower slightly
        grip = grip_close  # close grip
    elif stage == "lift":
        if is_first_call:
            saved_cube_pos = cube_pos
            is_first_call = False
        target_pos = np.array([saved_cube_pos[0], saved_cube_pos[1], 0.25])
        grip = grip_close  # keep closed
    elif stage == "to_box":
        target_pos = box_pos + np.array([0.0, 0.0, 0.25])
        grip = grip_close
    elif stage == "stabilize_box":
        target_pos = box_pos + np.array([0.0, 0.0, 0.25])
        grip = grip_close
    elif stage == "release":
        target_pos = box_pos + np.array([0.0, 0.0, 0.25])
        grip = grip_open
    elif stage == "drop":
        if saved_cube_pos is not None:
            target_pos = np.array([saved_cube_pos[0], saved_cube_pos[1], 0.25])
        else:
            target_pos = cube_pos + np.array([0.0, 0.0, 0.25])
        grip = grip_open
    else:
        raise ValueError(f"Unknown stage: {stage}")
    qpos = task.franka.inverse_kinematics(
        link=eef,
        pos=target_pos,
        quat=quat,
    ).cpu().numpy()
    qpos_arm = qpos[:-2]  # Franka has 7 arm joints + 2 finger joints
    action = np.concatenate([qpos_arm, grip])
    return action.astype(np.float32)

def initialize_dataset(env: GenesisEnv) -> LeRobotDataset:
    task = env.task
    height = env.observation_height
    width = env.observation_width
    dict_idx = 0
    dataset_path = f"datasets/{task}_{dict_idx}"
    while os.path.exists(f"datasets/{task}_{dict_idx}"):
        dict_idx += 1
        dataset_path = f"datasets/{task}_{dict_idx}"
    # env.observation_spaceの内容に基づいてfeaturesを定義
    features = {"action": {"dtype": "float32", "shape": (AGENT_DIM,), "names": joints_name}}
    for key, space in env.observation_space.spaces.items():
        if key == "observation.state":
            states_name = [
                "eef_pos_x", "eef_pos_y", "eef_pos_z",
                "eef_quat_w", "eef_quat_x", "eef_quat_y", "eef_quat_z",
                "grip_left", "grip_right",
            ]
            features[key] = {"dtype": "float32", "shape": (9,), "names": states_name}
        elif key.startswith("observation.images"):
            # すべての画像は3チャンネル
            features[key] = {"dtype": "video", "shape": (height, width, 3), "names": ("height", "width", "channels")}
    lerobot_dataset = LeRobotDataset.create(
        repo_id=None,
        fps=30,
        root=dataset_path,
        robot_type="franka",
        use_videos=True,
        features=features,
        # batch_encoding_size=10,
        batch_encoding_size=1,
    )
    return lerobot_dataset

def main(task, stage_dict, observation_height=480, observation_width=640, episode_num=1, show_viewer=False):
    env = GenesisEnv(task=task, observation_height=observation_height, observation_width=observation_width, show_viewer=show_viewer)
    dataset = initialize_dataset(env)
    episode_configs = build_balanced_episode_configs(task, episode_num)
    ep = 0
    while ep < episode_num:
        try:
            print(f"\n🎬 Starting episode {ep+1}")
            episode_config = episode_configs[ep]
            print(f"  Episode config: {episode_config}")
            env.reset(options=episode_config.get("reset_options"))
            obs_dict = {"action": []}
            for key in env.observation_space.spaces.keys():
                obs_dict[key] = []
            save_flag = False
            # reset後の初期観測を取得
            current_obs = env.get_obs()
            # reset後の初期観測を取得
            current_obs = env.get_obs()
            # ステージリストを作成
            stage_sequence = []
            # 通常のタスク
            for stage in stage_dict.keys():
                stage_sequence.append((stage, stage_dict[stage], None))

            for stage_name, steps, target_name in stage_sequence:
                print(f"  Stage: {stage_name} (Target: {target_name})")
                for t in range(steps):
                    action = expert_policy(env, stage_name, target_cube_name=target_name)
                    
                    # 先に現在の観測とアクションを保存（obs[t]とaction[t]のペア）
                    obs_dict["action"].append(action)
                    for key in current_obs.keys():
                        if key in obs_dict.keys():
                            obs_dict[key].append(current_obs[key])
                    
                    # アクションを実行して次の観測を取得
                    current_obs, reward, _, _, _ = env.step(action)
                    
                    if reward > 0:
                        save_flag = True
            if not save_flag:
                print(f"🚫 Skipping episode {ep+1}")
                continue
            print(f"✅ Saving episode {ep+1}")
            ep += 1
            for i in range(len(obs_dict["action"])):
                obs = {"task": env.get_task_description()}
                for key in obs_dict.keys():
                    if key.startswith("observation.images") and isinstance(obs_dict[key][i], Image.Image):
                        obs_dict[key][i] = np.array(obs_dict[key][i])
                    obs[key] = obs_dict[key][i]
                dataset.add_frame(obs)
            dataset.save_episode()
        except Exception as e:
            print(f"⚠️ Error occurred during episode {ep+1}: {e}")
            print("🔄 Retrying episode...")
            env.close()
            env = GenesisEnv(task=task, observation_height=observation_height, observation_width=observation_width, show_viewer=show_viewer)
            continue
    env.close()
if __name__ == "__main__":
    # datasetを作成したいタスクを指定

    task_candidates = [
        "normal",
    ]
    
    for task in task_candidates:
        stage_dict = {
            "hover": 100, # cubeの上に手を持っていく
            "stabilize": 40, # cubeの上で手を安定させる
            "grasp": 20, # cubeを掴む
            "lift": 50, # cubeを持ち上げる
            "to_box": 60, # cubeを箱の上に持っていく
            "stabilize_box": 20, # cubeを箱の上で安定させる
            "release": 60 # cubeを離す
        }
        
        main(episode_num=500, task=task, stage_dict=stage_dict, observation_height=224, observation_width=224, show_viewer=False)


# normal: 音は関係なく，赤，青，緑のCubeから指定された色のCubeを箱に入れるタスク
# normal-fix: 音は関係なく，赤色のCubeを箱に入れるタスク