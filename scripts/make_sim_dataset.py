import numpy as np
import os
import traceback
from PIL import Image
from roboenv.env.genesis_env import GenesisEnv
from roboenv.lerobot_dataset_utils import append_episode_summary, build_frame, create_lerobot_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

saved_cube_pos = None
is_first_call = True

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
    dict_idx = 0
    dataset_path = f"datasets/{env.task}_{dict_idx}"
    while os.path.exists(f"datasets/{env.task}_{dict_idx}"):
        dict_idx += 1
        dataset_path = f"datasets/{env.task}_{dict_idx}"
    return create_lerobot_dataset(dataset_path, env, include_rl_labels=True)

def main(task, stage_dict, observation_height=480, observation_width=640, episode_num=1, show_viewer=False):
    env = GenesisEnv(task=task, observation_height=observation_height, observation_width=observation_width, show_viewer=show_viewer)
    dataset = initialize_dataset(env)
    ep = 0
    while ep < episode_num:
        try:
            print(f"\n🎬 Starting episode {ep+1}")
            env.reset()
            obs_dict = {"action": []}
            reward_history = []
            done_history = []
            success_history = []
            for key in env.observation_space.spaces.keys():
                obs_dict[key] = []
            save_flag = False
            
            # reset後の初期観測を取得
            current_obs = env.get_obs()
            
            # reset後の初期観測を取得
            current_obs = env.get_obs()
            
            # ステージリストを作成
            stage_sequence = []
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
                    current_obs, reward, terminated, truncated, info = env.step(action)
                    success = bool(info.get("is_success", reward > 0))
                    done = bool(terminated or truncated)
                    reward_history.append(float(reward))
                    done_history.append(done)
                    success_history.append(success)

                    if success:
                        save_flag = True
                        break
                if save_flag:
                    break
            if not save_flag:
                print(f"🚫 Skipping episode {ep+1}")
                continue
            print(f"✅ Saving episode {ep+1}")
            ep += 1
            for i in range(len(obs_dict["action"])):
                frame_obs = {}
                for key in env.observation_space.spaces.keys():
                    value = obs_dict[key][i]
                    if key.startswith("observation.images") and isinstance(value, Image.Image):
                        value = np.array(value)
                    frame_obs[key] = value
                is_last = i == len(obs_dict["action"]) - 1
                dataset.add_frame(
                    build_frame(
                        numpy_observation=frame_obs,
                        action=obs_dict["action"][i],
                        task=env.get_task_description(),
                        reward=reward_history[i],
                        done=done_history[i] if i < len(done_history) else is_last,
                        success=success_history[i] if i < len(success_history) else is_last and save_flag,
                        episode_success=save_flag,
                    )
                )
            dataset.save_episode()
            append_episode_summary(
                dataset_root=dataset.root,
                episode_index=dataset.meta.total_episodes - 1,
                success=save_flag,
                episode_return=float(sum(reward_history)),
                episode_length=len(obs_dict["action"]),
                task=env.get_task_description(),
                metadata=env.get_episode_metadata(),
            )
        except Exception as e:
            print(f"⚠️ Error occurred during episode {ep+1}: {e}")
            traceback.print_exc()
            print("🔄 Retrying episode...")
            if dataset is not None and getattr(dataset, "episode_buffer", None) is not None:
                try:
                    dataset.clear_episode_buffer(delete_images=True)
                except Exception as clear_error:
                    print(f"⚠️ Failed to clear dataset episode buffer: {clear_error}")
            env.close()
            env = GenesisEnv(task=task, observation_height=observation_height, observation_width=observation_width, show_viewer=show_viewer)
            continue
    dataset.finalize()
    env.close()
if __name__ == "__main__":
    # datasetを作成したいタスクを指定
    task_candidates = [
        "normal-fix",
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
        
        main(episode_num=50, task=task, stage_dict=stage_dict, observation_height=224, observation_width=224, show_viewer=False)

# normal: 赤，青，緑のCubeから言語で指定された色のCubeを箱に入れるタスク
# normal-fix: 赤，青，緑のCubeから赤色のCubeを箱に入れるタスク
