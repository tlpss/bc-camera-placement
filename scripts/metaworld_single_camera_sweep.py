from dataclasses import dataclass, field
import os
import shutil
import datetime
import gymnasium
import wandb
from pathlib import Path
from camera_placement import DATA_DIR
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.common.envs.configs import EnvConfig
from lerobot.scripts.train import train
from camera_placement.lerobot_dataset_recorder import LeRobotDatasetRecorder, collect_demonstrations_non_blocking
from camera_placement.multiview_metaworld import  DEFAULT_CAMERAS_CONFIG, get_abs_policy_action, CameraConfig, ENV_POLICY_MAP
from camera_placement.metaworld_viewpoints import METAWORLD_CAMERA_VIEWPOINT_CONFIGS

CAMERA_SCENARIO_ID_TO_CONFIG = { i : METAWORLD_CAMERA_VIEWPOINT_CONFIGS[i] for i in range(len(METAWORLD_CAMERA_VIEWPOINT_CONFIGS))}
@dataclass
class Config:
    env_name: str = "assembly-v3"
    seed: int = 2025
    n_demonstrations: int = 10
    camera_scenario_id: int = 0 # wandb cannot handle specific combinations, so sweep over ID..


    n_steps: int = 100_000
    eval_freq: int = 50
    log_freq: int = 1000

    def wandb_name(self) -> str:
        return f"{self.env_name}-{self.camera_scenario_id}"


@EnvConfig.register_subclass("Meta-World")
@dataclass
class MetaWorldMultiviewConfig(EnvConfig):
    task:str = "multiview"
    metaworld_env: str = "reach-v3"
    fps: int = 80
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
            "abs_action_space": True,
        }
    


def run(config: Config):
    # Select the camera config based on scenario id
    camera_config = CAMERA_SCENARIO_ID_TO_CONFIG[config.camera_scenario_id]
    
    # Prepare environment config with single camera
    env_config = MetaWorldMultiviewConfig(
        metaworld_env=config.env_name,
        cameras_config=[camera_config],
    )

    # Prepare policy
    policy = ENV_POLICY_MAP[config.env_name]()
    def action_callable(env):
        return get_abs_policy_action(policy, env)

    # Use new output directory for datasets and artifacts
    base_data_dir = DATA_DIR / "metaworld" / "single-camera-sweep"
    os.makedirs(base_data_dir / "dataset", exist_ok=True)
    dataset_path = base_data_dir / "dataset" / datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Create environment and collect demonstrations
    env = gymnasium.make("Meta-World/multiview", **env_config.gym_kwargs)
    env.reset(seed=config.seed)
    dataset_recorder = LeRobotDatasetRecorder(env, dataset_path, f"tlips/metaworld-single", 80, True)
    collect_demonstrations_non_blocking(action_callable, env, dataset_recorder, n_episodes=config.n_demonstrations)

    # Build input/output features for single camera
    input_features = {
        f"observation.images.{camera_config.uid}": PolicyFeature(type=FeatureType.VISUAL, shape=(camera_config.height, camera_config.width, 3)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }

    train_config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="whatever/whatever", root=str(dataset_path)),
        policy=DiffusionConfig(
            push_to_hub=False,
            repo_id="whatever/whatever",
            n_obs_steps=2,
            n_action_steps=40,
            horizon=80,
            input_features=input_features,
            output_features=output_features,
            device='cuda',
            crop_shape=(84,112),
            crop_is_random=False,
            down_dims=(128,256,512),
            kernel_size=3,
            n_groups=8,
            diffusion_step_embed_dim=128,
            optimizer_lr=2e-4,
        ),
        eval=EvalConfig(n_episodes=20, batch_size=10),
        env=env_config,
        wandb=WandBConfig(enable=True, disable_artifact=True, project="camera-placement"),
        output_dir=base_data_dir / "outputs" / "train" / datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        job_name=f"metaworld-singlecam-sweep",
        resume=False,
        seed=config.seed,
        num_workers=4,
        batch_size=64,
        steps=config.n_steps,
        eval_freq=config.eval_freq,
        log_freq=config.log_freq,
        save_freq= 2*config.n_steps # don't save intermediate checkpoints
    )

    # Update train config from wandb sweep config if present
    wandb.config.update(vars(train_config))
    print(train_config)
    train(train_config)

    # Clean up dataset after training
    shutil.rmtree(dataset_path)
    # remove the output dir checkpoints
    if train_config.output_dir:
        shutil.rmtree(str(train_config.output_dir/ "checkpoints"))


if __name__ == "__main__":
    config = Config()
    wandb.init(project="camera-placement", name=f"1cam-sweep-{config.wandb_name()}", config=vars(config))
    config = Config(**wandb.config)
    # update wandb run name to include camera config
    wandb.run.name = f"1cam-sweep-{config.wandb_name()}"
    run(config)



