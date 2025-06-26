import gymnasium as gym
from pathlib import Path
import numpy as np
import torch

class LeRobotDatasetRecorder:
    """
    Gym dataset recorder for LeRobot.

    requires a gymnasium environment with a 'pixels' and 'agent_pos' key in the observation space.
    the 'pixels' key should be a dictionary of camera ids to numpy arrays of shape (H, W, 3).
    the 'agent_pos' key should be a numpy array of shape (N,)

    This is required for Lerobot to use the env during training for eval, as these keys are hardcoded.
    """
    DEFAULT_FEATURES = {
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.success": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
        "seed": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "timestamp": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
    }

    def __init__(self, env: gym.Env, root_dataset_dir: Path, dataset_name: str, fps: int, use_videos=True):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.root_dataset_dir = root_dataset_dir
        self.dataset_name = dataset_name
        self.fps = fps

        self._n_recorded_episodes = 0
        self.key_mapping_dict = {}

        features = self.DEFAULT_FEATURES.copy()
        # add images to features

        # uses the lerobot convention to map to 'observation.image' keys
        # and stores as video.

        assert isinstance(env.observation_space, gym.spaces.Dict), "Observation space should be a dict, got {}".format(type(env.observation_space))
        assert "pixels" in env.observation_space.keys(), "Observation space should contain 'pixels' key"
        self.image_keys = env.observation_space["pixels"].keys()

    
        for key in self.image_keys:
            shape = env.observation_space.spaces["pixels"][key].shape

            if not key.startswith("observation.images"):
                lerobot_key = f"observation.images.{key}"
                self.key_mapping_dict[key] = lerobot_key

            lerobot_key = self.key_mapping_dict.get(key, key)
            if "/" in lerobot_key:
                self.key_mapping_dict[key] = lerobot_key.replace("/", "_")
            lerobot_key = self.key_mapping_dict[key]
            if use_videos:
                features[lerobot_key] = {"dtype": "video", "names": ["height", "width", "channel"], "shape": shape}
            else:
                features[lerobot_key] = {"dtype": "image", "shape": shape, "names": None}

        # state observations
        if "agent_pos" in env.observation_space.keys():
            self.state_keys = ["agent_pos"]

        for key in self.state_keys:
            if "/" in key:
                self.key_mapping_dict[key] = key.replace("/", "_")
                lerobot_key = self.key_mapping_dict.get(key, key)
            else:
                lerobot_key = key
            shape = env.observation_space.spaces[key].shape
            features[lerobot_key] = {"dtype": "float32", "shape": shape, "names": None}

        # add single 'state' observation that concatenates all state observations
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (sum([env.observation_space.spaces[key].shape[0] for key in self.state_keys]),),
            "names": None,
        }
        # add action to features
        features["action"] = {"dtype": "float32", "shape": env.action_space.shape, "names": None}

        print(f"Features: {features}")
        # create the dataset
        self.lerobot_dataset = LeRobotDataset.create(
            repo_id=dataset_name,
            fps=self.fps,
            root=self.root_dataset_dir,
            features=features,
            use_videos=use_videos,
            image_writer_processes=0,
            image_writer_threads=8,
        )

    def start_episode(self):
        pass

    def record(self, obs, action, reward, done, info):

        frame = {
            "action": torch.from_numpy(action).float(),
            "next.reward": torch.tensor([reward]).float(),
            "next.success": torch.tensor([done]),
            "seed": torch.tensor([0]),  # TODO: store the seed
        }
        for key in self.image_keys:
            lerobot_key = self.key_mapping_dict.get(key, key)
            frame[lerobot_key] = obs["pixels"][key]

        for key in self.state_keys:
            lerobot_key = self.key_mapping_dict.get(key, key)
            frame[lerobot_key] = torch.from_numpy(obs[key]).float()

        # concatenate all 'state' observations into a single tensor
        state = torch.cat([frame[self.key_mapping_dict.get(key,key)].flatten() for key in self.state_keys])
        frame["observation.state"] = state

        self.lerobot_dataset.add_frame(frame,task="")

    def save_episode(self):
        self.lerobot_dataset.save_episode()
        self._n_recorded_episodes += 1

    def finish_recording(self):
        # computing statistics
        pass

    @property
    def n_recorded_episodes(self):
        return self._n_recorded_episodes


def collect_demonstrations_non_blocking(agent_callable, env, dataset_recorder, n_episodes=50):
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        dataset_recorder.start_episode()
        while not done:
            action = agent_callable(env)
            new_obs, reward, termination, truncation, info = env.step(action)
            done = termination or truncation
            dataset_recorder.record(obs, action, reward, done, info)
            obs = new_obs
        dataset_recorder.save_episode()

    dataset_recorder.finish_recording()

if __name__ == "__main__":

    import os
    import shutil
    from metaworld.policies.sawyer_button_press_topdown_wall_v3_policy import SawyerButtonPressTopdownWallV3Policy
    if os.path.exists("data/lerobot/multiview_test"):
        shutil.rmtree("data/lerobot/multiview_test")
    from camera_placement.multiview_metaworld import DEFAULT_CAMERA_CONFIGS

    policy = SawyerButtonPressTopdownWallV3Policy()
    env = gym.make("Meta-World/multiview", metaworld_env_name="button-press-topdown-wall-v3", camera_configs=DEFAULT_CAMERA_CONFIGS)

    dataset_recorder = LeRobotDatasetRecorder(env, Path("data/lerobot/multiview_test"), "multiview_test", 10, use_videos=True)

    def action_callable(env):
        return policy.get_action(env.unwrapped._get_obs())
    collect_demonstrations_non_blocking(action_callable, env, dataset_recorder, n_episodes=10)