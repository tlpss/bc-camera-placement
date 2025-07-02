import torch
import time
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import NormalizationMode, PolicyFeature, FeatureType

# ---- 1. Define minimal input/output features ----
# Example: state_dim=7, action_dim=4, n_obs_steps=2, n_action_steps=2, horizon=4
state_dim = 7
action_dim = 10
n_obs_steps = 2
n_action_steps = 80
horizon = 120
image_size = 280
n_cameras = 3
input_features = {
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    "observation.images.left": PolicyFeature(type=FeatureType.VISUAL, shape=(3,image_size, image_size)),
    "observation.images.right": PolicyFeature(type=FeatureType.VISUAL, shape=(3,image_size, image_size)),
    "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3,image_size, image_size)),
}
output_features = {
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))
}

# ---- 2. Create config and policy ----
config = DiffusionConfig(
    input_features=input_features,
    output_features=output_features,
    normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
    },
    n_obs_steps=n_obs_steps,
    n_action_steps=n_action_steps,
    horizon=horizon,
    crop_shape=None,
    num_inference_steps=10,
    noise_scheduler_type="DDIM",
)
policy = DiffusionPolicy(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy.to(device)

# ---- 3. Create dummy input batch for inference ----
batch_size = 1

dummy_batch = {
    "observation.state": torch.randn(batch_size, n_obs_steps, state_dim, device=device),
    "observation.images": torch.randn(batch_size,config.n_obs_steps, n_cameras, 3, image_size, image_size,  device=device),
}
# ---- 4. Warmup ----
with torch.no_grad():
    for _ in range(3):
        _ = policy.diffusion.generate_actions(dummy_batch)

# ---- 5. Measure forward pass during training ---- 
n_runs = 50
torch.cuda.synchronize() if device.type == "cuda" else None
start = time.time()
with torch.no_grad():
    for _ in range(n_runs):
        #_ = policy.forward(dummy_batch)
        _ = policy.diffusion.generate_actions(dummy_batch)
torch.cuda.synchronize() if device.type == "cuda" else None
end = time.time()
print(f"Average inference time for single observation: {(end - start) / n_runs * 1000:.2f} ms")


## ---- 6. Measure forward pass during training ---- 
batch_size = 64
dummy_batch = {
    "observation.state": torch.randn(batch_size, n_obs_steps, state_dim, device=device),
    "observation.images.left": torch.randn(batch_size,config.n_obs_steps, 3, image_size, image_size,  device=device),
    "observation.images.right": torch.randn(batch_size,config.n_obs_steps, 3, image_size, image_size,  device=device),
    "observation.images.wrist": torch.randn(batch_size,config.n_obs_steps, 3, image_size, image_size,  device=device),
    "action": torch.randn(batch_size, config.horizon, action_dim, device=device),
    "action_is_pad": torch.zeros(batch_size, config.horizon, device=device),
}
n_runs = 50
torch.cuda.synchronize() if device.type == "cuda" else None
start = time.time()
with torch.no_grad():
    for _ in range(n_runs):
        _ = policy.forward(dummy_batch)
torch.cuda.synchronize() if device.type == "cuda" else None
end = time.time()
print(f"Average forward pass time: {(end - start) / n_runs * 1000:.2f} ms")
