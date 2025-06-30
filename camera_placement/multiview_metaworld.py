from functools import partial
import gymnasium as gym
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import metaworld #noqa
from dataclasses import dataclass
from typing import List

from metaworld.policies import ENV_POLICY_MAP



@dataclass
class CameraConfig:
    """Configuration for a camera in the multiview environment."""
    uid: str
    distance: float
    azimuth: float
    elevation: float
    lookat: List[float]
    width: int
    height: int
    fovy: float = 45.0


class MetaWorldProprioceptiveStateWrapper(gym.Wrapper):
    """Wrapper that propagates only proprioceptive state information from MetaWorld environments."""
    
    def __init__(self, env):
        super().__init__(env)
        # Update observation space to only include agent position (first 4 elements)
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[:4], 
            high=self.env.observation_space.high[:4], 
            dtype=np.float64
        )

        # update render_fps of underlying env to 10 FPS
        self.unwrapped.metadata["render_fps"] = 10
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Transform observation to only include agent position
        obs = obs[:4]  # only add the proprioceptive state of the robot
        terminated = info["success"] == 1.0  # somehow MetaWorld never returns termination conditions, I use it differently.
        info["is_success"] = info["success"] == 1.0 # Lerobot expects this key for evaluation.
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Transform observation to only include agent position
        obs = obs[:4]  # only add the proprioceptive state of the robot
        return obs, info


class MultiViewMetaworldEnv(gym.Wrapper):
    def __init__(self, env, cameras_config: List[CameraConfig]):
        """
        Wrapper that renders arbitrary camera views for a MetaWorld environment
        
        Args:
            env: The MetaWorld gymnasium environment (should already have observation transformed)
            cameras_config: List of CameraConfig objects defining camera parameters
        """
        super().__init__(env)
        self.cameras_config = cameras_config
        self.renderers = {}
        self._init_renderers()

        # Update observation space to include both agent position and camera pixels
        self.observation_space = gym.spaces.Dict({
            "agent_pos": self.env.observation_space,  # Use the transformed observation space from underlying env
            "pixels": gym.spaces.Dict({config.uid: gym.spaces.Box(low=0, high=255, shape=(config.height, config.width, 3), dtype=np.uint8) for config in self.cameras_config})
        })
        
        
    def _init_renderers(self):
        # Create renderer for each camera config
        for config in self.cameras_config:
            renderer = MujocoRenderer(
                self.unwrapped.model,
                self.unwrapped.data, 
                {"distance": config.distance, "azimuth": config.azimuth, "elevation": config.elevation, "lookat": np.array(config.lookat)},
                width=config.width,
                height=config.height,
                camera_name=config.uid
            )
            self.renderers[config.uid] = renderer


    def _build_obs(self, state_obs):
        obs_dict = {
            "agent_pos": state_obs,  # Use the already transformed observation
        }
        pixel_dict = {}
        for uid, renderer in self.renderers.items():
            pixel_dict[uid] = renderer.render(render_mode="rgb_array")
        obs_dict["pixels"] = pixel_dict
        return obs_dict
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._build_obs(obs)
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._build_obs(obs)
        return obs, info
    
    def render(self):
        if not self.cameras_config:
            raise RuntimeError("No camera configs available for rendering.")
        first_uid = self.cameras_config[0].uid
        renderer = self.renderers[first_uid]
        return renderer.render(render_mode="rgb_array")



from metaworld import ALL_V3_ENVIRONMENTS

# register with gymnasium
def entry_point(**kwargs):
    env_id = kwargs.pop("metaworld_env_name",None)
    assert env_id in ALL_V3_ENVIRONMENTS, f"env_id must be one of {ALL_V3_ENVIRONMENTS}"
    cameras_config = kwargs.pop("cameras_config",None)
    if not env_id:
        raise ValueError("metaworld_env_name must be provided")
    if not cameras_config:
        raise ValueError("cameras_config must be provided")
    
    # Convert dictionaries to CameraConfig objects
    if isinstance(cameras_config[0], dict):
        cameras_config = [CameraConfig(**config) for config in cameras_config]
    
    # Create base environment
    env = gym.make("Meta-World/MT1", env_name=env_id,**kwargs)
    
    # Apply observation transformation wrapper first
    env = MetaWorldProprioceptiveStateWrapper(env)
    
    # Apply multiview wrapper
    env = MultiViewMetaworldEnv(env, cameras_config)
    
    return env

gym.register(
    id ="Meta-World/multiview",
    entry_point=entry_point,
)


DEFAULT_CAMERAS_CONFIG = [
    CameraConfig(
        uid="camera_1",
        distance=1.5,
        azimuth=-145,
        elevation=-30.0,
        lookat=[0, 0.65, 0.1],
        fovy=45,
        width=128,
        height=96,
    )
]




if __name__ == "__main__":
    # Example usage:
    camera_configs = [
        {
            "uid": "camera_1",
            "distance": 2.0,
            "azimuth": 215,
            "elevation": -20.0,
            "lookat": [0, 0.5, 0.0],
            "width": 200,
            "height": 200,
        },
        {
            "uid": "camera_2",
            "distance": 2.0,
            "azimuth": 90,
            "elevation": -45.0, 
            "lookat": [0, 0.5, 0.0],
            "width": 200,
            "height": 200,
        },
        {
            "uid": "camera_3",
            "distance": 2.5,
            "azimuth": 180,
            "elevation": -30.0,
            "lookat": [0, 0.5, 0.0],
            "width": 128,
            "height": 128,
        }
    ]

    env = gym.make("Meta-World/multiview", metaworld_env_name="reach-v3", cameras_config=camera_configs, seed=2025, render_mode="rgb_array", width=200, height=200)
    obs, info = env.reset()

    print(env.observation_space)
    obs, info = env.reset()


    # Now when you step the environment, you'll get renders from all cameras in info['camera_renders']
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    camera_renders = obs["pixels"]
    for uid, render in camera_renders.items():
        print(f"camera {uid} has shape {render.shape}")
    
    obs,reward,terminated,truncated,info = env.step(env.action_space.sample())
    new_camera_renders = obs["pixels"]
    # check if the new camera renders are different from the old ones
    for uid, render in new_camera_renders.items():
        if not np.allclose(render, camera_renders[uid]):
            print(f"camera {uid} has changed")
