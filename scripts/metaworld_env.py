import gymnasium as gym
import metaworld
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
SEED = 0  # some seed number here
# print all avaliable envs


env = gym.make('Meta-World/MT1', env_name='basketball-v3', seed=SEED, render_mode="rgb_array", width=200, height=200)
obs, info = env.reset()
DEFAULT_CAMERA_CONFIG = {
"distance": 1,
"azimuth": 215,
"elevation": -20.0,
"lookat": np.array([0, 0.5, 0.0]),
}


env.unwrapped.mujoco_renderer  = MujocoRenderer(env.unwrapped.model, env.unwrapped.data, DEFAULT_CAMERA_CONFIG, width=200, height=200)

img = env.render()

import PIL
from PIL import Image

Image.fromarray(img).save("metaworld_reach_v3.png")

class MultiViewWrapper(gym.Wrapper):
    def __init__(self, env, camera_configs):
        """
        Wrapper that renders multiple camera views for a MetaWorld environment
        
        Args:
            env: The MetaWorld gymnasium environment
            camera_configs: List of camera configuration dictionaries, each containing:
                - distance: float
                - azimuth: float 
                - elevation: float
                - lookat: np.array([x, y, z])
        """
        super().__init__(env)
        self.camera_configs = camera_configs
        self.renderers = []
        
        # Create renderer for each camera config
        for config in camera_configs:
            renderer = MujocoRenderer(
                env.unwrapped.model,
                env.unwrapped.data, 
                config,
                width=env.unwrapped.mujoco_renderer.width,
                height=env.unwrapped.mujoco_renderer.height
            )
            self.renderers.append(renderer)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Render from all cameras
        renders = []
        for renderer in self.renderers:
            self.unwrapped.mujoco_renderer = renderer
            renders.append(self.env.render())
            
        info['camera_renders'] = renders
        return obs, reward, terminated, truncated, info

# Example usage:
camera_configs = [
    {
        "distance": 2.0,
        "azimuth": 215,
        "elevation": -20.0,
        "lookat": np.array([0, 0.5, 0.0]),
    },
    {
        "distance": 2.0,
        "azimuth": 90,
        "elevation": -45.0, 
        "lookat": np.array([0, 0.5, 0.0]),
    },
    {
        "distance": 2.5,
        "azimuth": 180,
        "elevation": -30.0,
        "lookat": np.array([0, 0.5, 0.0]),
    }
]

env = MultiViewWrapper(env, camera_configs)
obs, info = env.reset()

# Now when you step the environment, you'll get renders from all cameras in info['camera_renders']
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
camera_renders = info['camera_renders']

# Save example images from each camera
for i, img in enumerate(camera_renders):
    Image.fromarray(img).save(f"camera_view_{i}.png")

# get expert trajectories
from metaworld.policies.sawyer_basketball_v3_policy import SawyerBasketballV3Policy
policy = SawyerBasketballV3Policy()

for j in range(200):
    obs, reward, terminated, truncated, info = env.step(policy.get_action(obs))
    camera_renders = info['camera_renders']
    for i, img in enumerate(camera_renders[:1]):
        Image.fromarray(img).save(f"camera_view_{j}_{i}.png")