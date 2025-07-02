# bc-camera-placement


## installation
- clone with submodules: `git clone --recurse-submodules https://github.com/tlips/bc-camera-placement.git`
- install uv: `wget -qO- https://astral.sh/uv/install.sh | sh` 
- build uv venv : `uv sync --prerelease=allow`
- add package in editable mode `uv pip install -e .`
- run `export MUJOCO_GL=egl` for headless rendering.


## VM
- login on wandb `uv run --prerelease=allow wandb login`
- install screen `sudo apt-get update && sudo apt-get install -y screen`
- libmesa (egl) `sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libegl1-mesa`