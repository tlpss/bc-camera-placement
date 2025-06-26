from dataclasses import dataclass, field
import multiprocessing
from pathlib import Path
from lerobot.common.envs.configs import EnvConfig
from lerobot.common.utils.utils import init_logging
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.scripts.train import train
import gymnasium

from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy
from camera_placement.lerobot_dataset_recorder import LeRobotDatasetRecorder, collect_demonstrations_non_blocking
from camera_placement.multiview_metaworld import DEFAULT_CAMERA_CONFIGS

import os 
import shutil
import datetime

# if os.path.exists("tmp/dataset/robot_push_button"):
#     shutil.rmtree("tmp/dataset/robot_push_button")

# if os.path.exists("tmp/outputs/train"):
#     shutil.rmtree("tmp/outputs/train")

 ## training
@EnvConfig.register_subclass("Meta-World")
@dataclass
class MetaWorldMultiviewConfig(EnvConfig):
    task:str = "multiview"
    metaworld_env: str = "reach-v3"
    fps: int = 10
    max_episode_steps: int = 100
    camera_configs: list[dict] = field(default_factory=lambda: DEFAULT_CAMERA_CONFIGS)
    
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": "action",
            "agent_pos": "observation.state",
        }
    )

    def __post_init__(self):
        for camera_config in self.camera_configs:
            self.features[f"pixels/{camera_config['uid']}"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3,camera_config["width"], camera_config["height"]))
            self.features_map[f"pixels/{camera_config['uid']}"] = f"observation.images.{camera_config['uid']}"
    
    
    @property
    def gym_kwargs(self) -> dict:
        return {
            "metaworld_env_name": self.metaworld_env,
            "camera_configs": self.camera_configs,
            "render_mode": "rgb_array",
            "max_episode_steps": self.max_episode_steps,
        }
    

def run(seed,env,env_config: EnvConfig,demonstrator_callable):
    os.makedirs("tmp/dataset/mwmv", exist_ok=True)
    dataset_path = f"tmp/dataset/mwmv/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    ## create dataset and store

  
    env.reset()
    dataset_recorder = LeRobotDatasetRecorder(env,dataset_path,f"tlips/metaworld-mv",10,True)
    collect_demonstrations_non_blocking(demonstrator_callable, env, dataset_recorder, n_episodes=2)

    camera_config=env_config.camera_configs[0],
    input_features = {}
    for camera_config in env_config.camera_configs:
        input_features[f"observation.images.{camera_config['uid']}"] = PolicyFeature(type=FeatureType.VISUAL, shape=(camera_config["height"],camera_config["width"], 3))
    input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))
    input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }

    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="whatever/whatever", root=dataset_path),
        policy=ACTConfig(
            device='cpu',
            n_obs_steps=1,
            chunk_size=20,
            n_action_steps=10,
            input_features=input_features,
            output_features=output_features,
            dim_feedforward=1024,
            dim_model=512,
            optimizer_lr=2e-5,

            ),

        eval=EvalConfig(n_episodes=20, batch_size=10),
        env=env_config,
        wandb=WandBConfig(enable=True, disable_artifact=True,project="camera-placement"),
        output_dir=Path(f"tmp/outputs/train/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"), 
        job_name=f"metaworld-{env_config.task}-cam-az={camera_config['azimuth']}-el={camera_config['elevation']}-dist={camera_config['distance']}",
        resume=False,
        seed=seed,
        num_workers=4,
        batch_size=64,
        steps=100_000,
        eval_freq=100,
        log_freq=100,
        save_freq=100_000
    )

    # train policy

    print(config)
    train(config)


# load the latest checkpoint and eval further?


if __name__ == "__main__":
    import numpy as np 
    look_at_position = [0.0, 0.7, 0.1] # approx average of object spawn space for many tasks 

    camera_configs = [
        [
            {
                "uid": "scene",
                "width": 256,
                "height": 256,
                "lookat": look_at_position,
                "distance": 0.6,
                "azimuth": 0,
                "elevation": 45,
                "fovy": 45,
            }
        ]
    ]

    seed = 2025
    env_name = "reach-v3"
    policy = SawyerReachV3Policy()
    def action_callable(env):
        return policy.get_action(env.unwrapped._get_obs())
    

    # create new wandb run
    import wandb 
    # check if run is active, if so, finish it
    for seed in [2025]: #,2026,2027]:
        for camera_position_config in camera_configs:
            env_config = MetaWorldMultiviewConfig(
                metaworld_env=env_name,
                camera_configs=camera_position_config,
            )
            env = gymnasium.make("Meta-World/multiview", **env_config.gym_kwargs)
            env.reset(seed=seed)
            if wandb.run is not None:
                print("Wandb run is active, finishing it...")
                wandb.finish()
            print(f"Running for env: {env_name} with camera config: {camera_position_config} and seed: {seed}")
            init_logging()


            run(seed,env,env_config,action_callable)