from copy import deepcopy
from camera_placement.multiview_metaworld import CameraConfig


look_at_position = [0.0, 0.65, 0.0] # approx average of object spawn space for many tasks 

y_resolution = 96
x_resolution = int(y_resolution*4/3) # 4:3 aspect ratio as used in Diffusion Policy paper and ACT.

top_down_camera_config = CameraConfig(
    uid="top_down_camera",
    distance=1.5,
    azimuth=-90,
    elevation=-89,
    lookat=look_at_position,
    height=y_resolution,
    width=x_resolution,
)

low_camera_config = CameraConfig(
    uid="low_front_camera",
    distance=.9,
    azimuth=-90,
    elevation=-20,
    lookat=look_at_position,
    height=y_resolution,
    width=x_resolution,
)

low_far_camera_config = CameraConfig(
    uid="low_front_camera",
    distance=1.5,
    azimuth=-90,
    elevation=-20,
    lookat=look_at_position,
    height=y_resolution,
    width=x_resolution,
)

mid_camera_config = CameraConfig(
    uid="mid_camera",
    distance=.9,
    azimuth=-90,
    elevation=-45,
    lookat=look_at_position,
    height=y_resolution,
    width=x_resolution,
)

mid_far_camera_config = CameraConfig(
    uid="mid_far_camera",
    distance=1.5,
    azimuth=-90,
    elevation=-45,
    lookat=look_at_position,
    height=y_resolution,
    width=x_resolution,
)

high_camera_config = CameraConfig(
    uid="high_camera",
    distance=0.9,
    azimuth=-90,
    elevation=-60,
    lookat=look_at_position,
    height=y_resolution,
    width=x_resolution,
)

high_far_camera_config = CameraConfig(
    uid="high_far_camera",
    distance=1.5,
    azimuth=-90,
    elevation=-60,
    lookat=look_at_position,
    height=y_resolution,
    width=x_resolution,
)


camera_configs = [top_down_camera_config, low_camera_config, low_far_camera_config, mid_camera_config, mid_far_camera_config, high_camera_config, high_far_camera_config]


METAWORLD_CAMERA_VIEWPOINT_CONFIGS = [top_down_camera_config]
for config in camera_configs[1:]:
    for azimuth in [0, 45, 135, 180, 225, 270, 315]:
        config_copy = deepcopy(config)
        config_copy.azimuth = azimuth
        METAWORLD_CAMERA_VIEWPOINT_CONFIGS.append(config_copy)

if __name__ == "__main__":
    print(f"Total viewpoints: {len(METAWORLD_CAMERA_VIEWPOINT_CONFIGS)}")