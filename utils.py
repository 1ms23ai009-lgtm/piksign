"""Shared utilities for adversarial attack and text generation models."""

import base64
import hashlib
import json
import os
import random
from typing import Any, Dict, List, Union

import yaml
from omegaconf import OmegaConf

import wandb
from config_schema import MainConfig


def load_api_keys() -> Dict[str, List[str]]:
    """Load API keys from the api_keys file.

    Returns:
        Dict[str, List[str]]: Dictionary containing API keys for different models

    Raises:
        FileNotFoundError: If no api_keys file is found
    """
    for ext in ["yaml", "yml", "json"]:
        file_path = f"api_keys.{ext}"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                if ext in ["yaml", "yml"]:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)

    raise FileNotFoundError(
        "API keys file not found. Please create api_keys.yaml, api_keys.yml, or api_keys.json "
        "in the root directory with your API keys."
    )


def get_api_keys(model_name: str) -> List[str]:
    """Get all API keys for specified model.

    Args:
        model_name: Name of the model to get API keys for

    Returns:
        List[str]: List of all API keys for the specified model

    Raises:
        KeyError: If API keys for model are not found
    """
    api_keys = load_api_keys()
    if model_name not in api_keys:
        raise KeyError(
            f"API key for {model_name} not found in api_keys file. "
            f"Available models: {list(api_keys.keys())}"
        )

    keys_list = api_keys[model_name]
    if not keys_list:
        raise ValueError(f"No API keys available for {model_name}")

    return keys_list


def get_api_key_count(model_name: str) -> int:
    """Get the number of available API keys for a model.

    Args:
        model_name: Name of the model

    Returns:
        int: Number of available API keys

    Raises:
        KeyError: If API keys for model are not found
    """
    return len(get_api_keys(model_name))


def hash_training_config(cfg: MainConfig) -> str:
    """Create a deterministic hash of training-relevant config parameters.

    Args:
        cfg: Configuration object containing model settings

    Returns:
        str: MD5 hash of the config parameters
    """
    # sepcial case, use only in the rebuttal
    if hasattr(cfg, "ata_noise"):
        noise_val = cfg.ata_noise
        return f'ata_noise_{noise_val}'
    # Convert backbone list to plain Python list
    if isinstance(cfg.model.backbone, (list, tuple)):
        backbone = list(cfg.model.backbone)
    else:
        backbone = OmegaConf.to_container(cfg.model.backbone)

    # Create config dict with converted values
    train_config = {
        "data": {
            "batch_size": int(cfg.data.batch_size),
            "num_samples": int(cfg.data.num_samples),
            "cle_data_path": str(cfg.data.cle_data_path),
            "tgt_data_path": str(cfg.data.tgt_data_path),
        },
        "optim": {
            "alpha": float(cfg.optim.alpha),
            "epsilon": int(cfg.optim.epsilon),
            "steps": int(cfg.optim.steps),
            "optimizer": str(cfg.optim.optimizer),
            "momentum": float(cfg.optim.momentum),
            "momentum_decay": float(cfg.optim.momentum_decay),
            "align": str(cfg.optim.align),
            "tm_idx": list(cfg.optim.tm_idx),
            "beta": float(cfg.optim.beta),
            "use_retrieval": bool(cfg.optim.use_retrieval),
        },
        "model": {
            "input_res": int(cfg.model.input_res),
            "crop_scale_1": tuple(float(x) for x in cfg.model.crop_scale_1),
            "crop_scale_2": tuple(float(x) for x in cfg.model.crop_scale_2),
            "crop_scale_3": tuple(float(x) for x in cfg.model.crop_scale_3),
            "changer_iters": list(cfg.model.changer_iters),
            "ensemble": bool(cfg.model.ensemble),
            "backbone": backbone,
            "target_num": int(cfg.model.target_num),
            "target_crop": bool(cfg.model.target_crop),
        },
        "attack": cfg.attack,
    }

    # Conditionally add multi_pass_num if attack is multi-pass
    if cfg.attack == "pgd_multi_pass":  # Add other multi-pass attack types if needed
        train_config["optim"]["multi_pass_num"] = int(cfg.optim.multi_pass_num)

    # Convert to JSON string with sorted keys
    json_str = json.dumps(train_config, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()


def setup_wandb(cfg: MainConfig, tags=None, name: str = None) -> None:
    """Initialize Weights & Biases logging.

    Args:
        cfg: Configuration object containing wandb settings
        tags: Optional list of tags for the run.
        name: Optional name for the run.
    """
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=config_dict,
        tags=tags,
        name=name,
        reinit=True,
    )


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string.

    Args:
        image_path: Path to image file

    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path to ensure exists
    """
    os.makedirs(path, exist_ok=True)


def get_output_paths(cfg: MainConfig, config_hash: str) -> Dict[str, str]:
    """Get dictionary of output paths based on config.

    Args:
        cfg: Configuration object
        config_hash: Hash of training config

    Returns:
        Dict[str, str]: Dictionary containing output paths
    """
    return {
        "output_dir": os.path.join(cfg.data.output, "img", config_hash),
        "desc_output_dir": os.path.join(cfg.data.output, "description", config_hash),
    }


# Create batches from a list
def create_batches(items, batch_size):
    """Split a list into batches of specified size.

    Args:
        items: List of items to split
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def log_metrics(pbar, metrics, img_index, epoch=None, log_wandb=True):
    """
    Log metrics to progress bar and wandb.

    Args:
        pbar: tqdm progress bar to update
        metrics: Dictionary of metrics to log
        img_index: Index of the image (for wandb logging)
        epoch: Optional epoch number for logging
        log_wandb: Whether to log to wandb
    """
    # Format metrics for progress bar
    pbar_metrics = {
        k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" for k, v in metrics.items()
    }
    pbar.set_postfix(pbar_metrics)

    # Prepare wandb metrics with image index
    wandb_metrics = {f"img{img_index}_{k}": v for k, v in metrics.items()}
    if epoch is not None:
        wandb_metrics["epoch"] = epoch

    # Log to wandb
    if log_wandb:
        wandb.log(wandb_metrics)
