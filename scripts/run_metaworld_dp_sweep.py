#!/usr/bin/env python3
"""
Script to run wandb sweep for MetaWorld diffusion policy hyperparameter search.
"""

import yaml
import wandb
from pathlib import Path


def load_sweep_config(yaml_path: str) -> dict:
    """Load sweep configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def run_sweep():
    """Run the wandb sweep for diffusion policy hyperparameter search."""
    
    # Load sweep configuration
    sweep_config = load_sweep_config("scripts/metaworld_dp_sweep.yaml")
    
    
    # Create sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project="camera-placement", 
    )
    
    print(f"Sweep created with ID: {sweep_id}")
    
    # # Start the sweep agent
    # wandb.agent(sweep_id, function=lambda: None, count=50)  # Run 50 trials
    
    # print("Sweep completed!")


if __name__ == "__main__":
    run_sweep() 