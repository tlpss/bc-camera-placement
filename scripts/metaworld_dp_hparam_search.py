from dataclasses import dataclass, field
import multiprocessing
from pathlib import Path
from lerobot.common.envs.configs import EnvConfig
from lerobot.common.utils.utils import init_logging
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.scripts.train import train
import gymnasium

from camera_placement.lerobot_dataset_recorder import LeRobotDatasetRecorder, collect_demonstrations_non_blocking
from camera_placement.multiview_metaworld import DEFAULT_CAMERAS_CONFIG, ENV_POLICY_MAP, CameraConfig

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
    max_episode_steps: int = 200
    cameras_config: list[CameraConfig] = field(default_factory=lambda: DEFAULT_CAMERAS_CONFIG)
    
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
        for camera_config in self.cameras_config:
            self.features[f"pixels/{camera_config.uid}"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3,camera_config.width, camera_config.height))
            self.features_map[f"pixels/{camera_config.uid}"] = f"observation.images.{camera_config.uid}"
    
    
    @property
    def gym_kwargs(self) -> dict:
        return {
            "metaworld_env_name": self.metaworld_env,
            "cameras_config": self.cameras_config,
            "render_mode": "rgb_array",
            "max_episode_steps": self.max_episode_steps,
        }
    

@dataclass
class Config:
    env_name: str = "reach-v3"
    seed: int = 2025
    n_demonstrations: int = 150

    dp_action_steps: int = 8
    dp_horizon: int = 16
    dp_down_dims: tuple[int, ...] = (512, 1024, 2048)
    dp_kernel_size: int = 5
    dp_n_groups: int = 8
    dp_diffusion_step_embed_dim: int = 128
    dp_optimizer_lr: float = 1e-4

    n_steps: int = 100_000
    eval_freq: int = 5000
    log_freq: int = 1000


def run(config: Config):

    policy = ENV_POLICY_MAP[config.env_name]()
    def action_callable(env):
        return policy.get_action(env.unwrapped._get_obs())
    
    
    os.makedirs("tmp/dataset/mwmv", exist_ok=True)
    dataset_path = Path(f"tmp/dataset/mwmv/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    ## create dataset and store
    env_config = MetaWorldMultiviewConfig(
        metaworld_env=config.env_name,
    )

    env = gymnasium.make("Meta-World/multiview", **env_config.gym_kwargs)
    env.reset(seed=config.seed)

    dataset_recorder = LeRobotDatasetRecorder(env, dataset_path, f"tlips/metaworld-mv", 10, True)
    collect_demonstrations_non_blocking(action_callable, env, dataset_recorder, n_episodes=config.n_demonstrations)

    input_features = {}
    for camera_config in env_config.cameras_config:
        input_features[f"observation.images.{camera_config.uid}"] = PolicyFeature(type=FeatureType.VISUAL, shape=(camera_config.height, camera_config.width, 3))
    input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }

    train_config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="whatever/whatever", root=str(dataset_path)),
        policy = DiffusionConfig(
            push_to_hub=False,
            repo_id="whatever/whatever",
            n_obs_steps=2,
            n_action_steps=config.dp_action_steps,
            horizon=config.dp_horizon,
            input_features=input_features,
            output_features=output_features,
            device='cuda',
            crop_shape=(84,112),
            crop_is_random=False,
            down_dims=config.dp_down_dims,
            kernel_size=config.dp_kernel_size,
            n_groups=config.dp_n_groups,
            diffusion_step_embed_dim=config.dp_diffusion_step_embed_dim,
            optimizer_lr=config.dp_optimizer_lr,
            
            
        ),
        eval=EvalConfig(n_episodes=20, batch_size=10),
        env=env_config,
        wandb=WandBConfig(enable=True, disable_artifact=True,project="camera-placement"),
        output_dir=Path(f"tmp/outputs/train/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"), 
        job_name=f"metaworld-sweep",
        resume=False,
        seed=config.seed,
        num_workers=4,
        batch_size=64,
        steps=config.n_steps,
        eval_freq=config.eval_freq,
        log_freq=config.log_freq,
        save_freq=config.n_steps
    )

    # train policy
    wandb.config.update(vars(train_config))
    print(train_config)
    train(train_config)

    # remove temp dataset
    shutil.rmtree(dataset_path)



if __name__ == "__main__":

    # create new wandb run
    config = Config()
    import wandb 
    wandb.init(project="camera-placement", name="metaworld-DP-sweep", config=vars(config))
    # updated config with wandb config in case of sweep
    config = Config(**wandb.config)
    run(config)