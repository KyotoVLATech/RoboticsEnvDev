from lerobot.common.envs import EnvConfig
from dataclasses import dataclass

@EnvConfig.register_subclass("test")
@dataclass
class TestEnv(EnvConfig):
    task: str = "test"
    fps: int = 30
    episode_length: int = 500
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    
    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }