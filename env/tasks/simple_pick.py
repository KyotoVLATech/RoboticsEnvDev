import genesis as gs
import numpy as np
from gymnasium import spaces
import random
import torch

joints_name = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "finger_joint1",
    "finger_joint2",
)
AGENT_DIM = len(joints_name)

class SimplePickTask:
    def __init__(self, observation_height, observation_width, show_viewer=False):
        self.show_viewer = show_viewer
        self.observation_height = observation_height
        self.observation_width = observation_width
        self._random = np.random.RandomState()
        self.box_scale = 1.0
        self._build_scene(show_viewer)
        self.observation_space = self._make_obs_space()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(AGENT_DIM,), dtype=np.float32)

    def _build_scene(self, show_viewer):
        if not gs._initialized:
            print("Genesis is not initialized, initializing now...")
            gs.init(backend=gs.cpu, precision="32", debug=False, logging_level="WARNING")
        # シーンを初期化
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(self.observation_width, self.observation_height),
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=show_viewer,
        )
        # 平面を追加
        self.plane = self.scene.add_entity(morph=gs.morphs.Plane())
        # フランカロボットを追加
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        # キューブAを追加
        self.cubeA = self.scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.65, 0.0, 0.025)),
            surface=gs.surfaces.Aluminium(color=(0.7, 0.3, 0.3)) # Red
        )
        # キューブBを追加
        self.cubeB = self.scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.35, 0.0, 0.025)),
            surface=gs.surfaces.Aluminium(color=(0.3, 0.3, 0.7)) # Blue
        )
        # キューブCを追加
        self.cubeC = self.scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.5, 0.0, 0.025)),
            surface=gs.surfaces.Aluminium(color=(0.3, 0.7, 0.3)) # Green
        )
        # フロントカメラを追加
        self.front_cam = self.scene.add_camera(
            res=(self.observation_width, self.observation_height),
            pos=(2.0, 0.0, 1.5),
            lookat=(0.5, 0.0, 0.2),
            fov=25,
            GUI=False
        )
        # サイドカメラを追加
        self.side_cam = self.scene.add_camera(
            res=(self.observation_width, self.observation_height),
            pos=(0.5, 1.5, 0.5),
            lookat=(0.5, 0.0, 0.1),
            fov=20,
            GUI=False
        )
        self.scene.build()
        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        self.eef = self.franka.get_link("hand")

    def _make_obs_space(self):
        return spaces.Dict({
            "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(AGENT_DIM,), dtype=np.float32),
            "observation.images.front": spaces.Box(low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8),
            "observation.images.side": spaces.Box(low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8),
        })
    
    def set_random_state(self, target, x_range, y_range, z):
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        z = z
        pos_tensor = torch.tensor([x, y, z], dtype=torch.float32, device=gs.device)
        quat_tensor = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=gs.device)
        target.set_pos(pos_tensor)
        target.set_quat(quat_tensor)
    
    def reset(self):
        self.color = random.choice(["red", "blue", "green"])
        # CubeAの位置をランダムに設定
        self.set_random_state(self.cubeA, (0.3, 0.7), (-0.3, 0.3), 0.04)
        # CubeBの位置をランダムに設定
        self.set_random_state(self.cubeB, (0.3, 0.7), (-0.3, 0.3), 0.04)
        # CubeCの位置をランダムに設定
        self.set_random_state(self.cubeC, (0.3, 0.7), (-0.3, 0.3), 0.04)
        # フランカロボットを初期位置にリセット
        qpos = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8, 0.04, 0.04])
        qpos_tensor = torch.tensor(qpos, dtype=torch.float32, device=gs.device)
        self.franka.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        self.franka.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
        )
        self.franka.set_qpos(qpos_tensor, zero_velocity=True)
        self.franka.control_dofs_position(qpos_tensor[:7], self.motors_dof)
        self.franka.control_dofs_position(qpos_tensor[7:], self.fingers_dof)

        # ステップ実行
        self.scene.step()
        self.front_cam.start_recording()
        self.side_cam.start_recording()
        return self.get_obs(), {}
        
    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self._random = np.random.RandomState(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.action_space.seed(seed)

    def step(self, action):
        action_tensor = torch.tensor(action, dtype=torch.float32, device=gs.device)
        self.franka.control_dofs_position(action_tensor[:7], self.motors_dof)
        self.franka.control_dofs_position(action_tensor[7:], self.fingers_dof)
        self.scene.step()
        reward = self.compute_reward()
        obs = self.get_obs()
        terminated = True if reward == 1.0 else False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
    
    def compute_reward(self):
        if self.color == "red":
            pos = self.cubeA.get_pos().cpu().numpy()
        elif self.color == "blue":
            pos = self.cubeB.get_pos().cpu().numpy()
        elif self.color == "green":
            pos = self.cubeC.get_pos().cpu().numpy()
        else:
            raise ValueError(f"Invalid color: {self.color}. Choose from 'red', 'blue', or 'green'.")
        # posとself.effの距離に基づいて報酬を計算
        eef_pos = self.eef.get_pos().cpu().numpy()
        distance = np.linalg.norm(eef_pos - pos)
        reward = 0.5 * np.exp(-distance)
        # posの高さに基づいて報酬を計算
        height = pos[2] - 0.025  # キューブの高さ
        reward += 0.5 * (1 - np.exp(-height))
        return reward

    def get_obs(self):
        # ロボットの状態を取得
        eef_pos = self.eef.get_pos().cpu().numpy()
        eef_rot = self.eef.get_quat().cpu().numpy()
        gripper = self.franka.get_dofs_position()[7:9].cpu().numpy()
        agent_pos = np.concatenate([eef_pos, eef_rot, gripper])
        # frontカメラの画像を取得
        front_pixels = self.front_cam.render()[0]
        assert front_pixels.ndim == 3, f"front_pixels shape {front_pixels.shape} is not 3D (H, W, 3)"
        # sideカメラの画像を取得
        side_pixels = self.side_cam.render()[0]
        assert side_pixels.ndim == 3, f"side_pixels shape {side_pixels.shape} is not 3D (H, W, 3)"
        obs = {
            "agent_pos": agent_pos,
            "observation.images.front": front_pixels,
            "observation.images.side": side_pixels,
        }
        return obs

    def save_videos(self, file_name, fps=30):
        self.front_cam.stop_recording(save_to_filename=f"{file_name}_front.mp4", fps=fps)
        self.side_cam.stop_recording(save_to_filename=f"{file_name}_side.mp4", fps=fps)

    def close(self):
        gs.destroy()

if __name__ == "__main__":
    import cv2
    gs.init(backend=gs.cpu, precision="32")
    task = SimplePickTask(observation_height=512, observation_width=512, show_viewer=True)
    task.reset()
    for _ in range(100):
        action = np.random.uniform(-1.0, 1.0, size=(AGENT_DIM,))
        task.step(action)