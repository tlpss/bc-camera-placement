from functools import partial
import gymnasium as gym
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import metaworld #noqa

class MultiViewMetaworldEnv(gym.Wrapper):
    def __init__(self, env, camera_configs):
        """
        Wrapper that renders arbitrarycamera views for a MetaWorld environment
        
        Args:
            env: The MetaWorld gymnasium environment
            camera_configs: List of camera configuration dictionaries, each containing:
                - uid: str
                - distance: float
                - azimuth: float 
                - elevation: float
                - lookat: np.array([x, y, z])
                - width: int
                - height: int
        """
        super().__init__(env)
        self.camera_configs = camera_configs
        self.renderers = {}
        self._init_renderers()

        #update observation space
        self.observation_space = gym.spaces.Dict({
            "agent_pos": gym.spaces.Box(low=self.env.observation_space.low[:4], high=self.env.observation_space.high[:4], dtype=np.float64),
            "pixels": gym.spaces.Dict({config["uid"]: gym.spaces.Box(low=0, high=255, shape=(config["height"], config["width"], 3), dtype=np.uint8) for config in self.camera_configs})
        })
        
        
    def _init_renderers(self):
        # Create renderer for each camera config
        for config in self.camera_configs:
            renderer = MujocoRenderer(
                self.unwrapped.model,
                self.unwrapped.data, 
                {"distance": config["distance"], "azimuth": config["azimuth"], "elevation": config["elevation"], "lookat": config["lookat"]},
                width=config["width"],
                height=config["height"],
                camera_name=config["uid"]
            )
            self.renderers[config["uid"]] = renderer


    def _build_obs(self,state_obs):
        
        obs_dict = {
            "agent_pos": state_obs[:4], # only add the proprioceptive state of the robot. #https://metaworld.farama.org/benchmark/state_space/
        }
        #self._init_renderers()
        pixel_dict = {}
        for uid, renderer in self.renderers.items():
            # update model and data
            # renderer.model = self.env.unwrapped.model
            # renderer.data = self.env.unwrapped.data
            pixel_dict[uid] = renderer.render(render_mode="rgb_array")
        obs_dict["pixels"] = pixel_dict
        return obs_dict
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        terminated = info["success"] == 1.0 # somehow MetaWorld never rerturns termination conditions,  I use it differently.
        obs = self._build_obs(obs)
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._build_obs(obs)
        return obs, info
    
    def render(self, mode="rgb_array"):
        # todo: build an overview scene camera and render that
        raise NotImplementedError("Render not implemented for MultiViewMetaworldEnv")


from metaworld import ALL_V3_ENVIRONMENTS

# register with gymnasium
def entry_point(**kwargs):
    env_id = kwargs.pop("metaworld_env_name",None)
    assert env_id in ALL_V3_ENVIRONMENTS, f"env_id must be one of {ALL_V3_ENVIRONMENTS}"
    camera_configs = kwargs.pop("camera_configs",None)
    if not env_id:
        raise ValueError("metaworld_env_name must be provided")
    if not camera_configs:
        raise ValueError("camera_configs must be provided")
    env = gym.make("Meta-World/MT1", env_name=env_id,**kwargs)
    return MultiViewMetaworldEnv(env, camera_configs)

gym.register(
    id ="Meta-World/multiview",
    entry_point=entry_point,
)


DEFAULT_CAMERA_CONFIGS = [
    {
        "uid": "camera_1",
        "distance": 2.0,
        "azimuth": 215,
        "elevation": -20.0,
        "lookat": np.array([0, 0.5, 0.0]),
        "fovy": 45,
        "width": 128,
        "height": 128,
    }
]




if __name__ == "__main__":
    # Example usage:
    camera_configs = [
        {
            "uid": "camera_1",
        "distance": 2.0,
        "azimuth": 215,
        "elevation": -20.0,
        "lookat": np.array([0, 0.5, 0.0]),
        "width": 200,
        "height": 200,
    },
    {
        "uid": "camera_2",
        "distance": 2.0,
        "azimuth": 90,
        "elevation": -45.0, 
        "lookat": np.array([0, 0.5, 0.0]),
        "width": 200,
        "height": 200,
    },
    {
        "uid": "camera_3",
        "distance": 2.5,
        "azimuth": 180,
        "elevation": -30.0,
        "lookat": np.array([0, 0.5, 0.0]),
        "width": 128,
        "height": 128,
    }
    ]

    env = gym.make("Meta-World/multiview", metaworld_env_name="reach-v3", camera_configs=camera_configs, seed=2025, render_mode="rgb_array", width=200, height=200)
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
