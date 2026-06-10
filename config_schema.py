from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class WandbConfig:
    """Wandb-specific configuration"""

    entity: str = "???"  # fill your wandb entity
    project: str = "your_wandb_project"
    run_name_prefix: str = ""  # Add a prefix to wandb run names, e.g., "dev" or "exp1"


@dataclass
class BlackboxConfig:
    """Configuration for blackbox model evaluation"""

    model_name: List[str] = field(
        default_factory=lambda: ["gpt5-thinking-low", "claude4.0", "gemini2.5pro"]
    )  # model aliases are implemented in blackbox_text_generation.py
    batch_size: int = 1
    timeout: int = 30
    parallel_images: int = 1  # Number of images to process in parallel


@dataclass
class DataConfig:
    """Data loading configuration"""

    batch_size: int = 1
    num_samples: int = 100
    cle_data_path: str = "resources/images/bigscale"
    tgt_data_path: str = "resources/images/target_images"
    output: str = "./outputs"
    retrieval_path: str = "resources/retrieved_embeddings"
    retrieval_dataset: str = "coco"
    retrieval_device: str = "auto"


@dataclass
class OptimConfig:
    """Optimization parameters"""

    alpha: float = 1.0
    epsilon: int = 8
    steps: int = 300
    optimizer: str = "adam"
    momentum: float = 0.9
    momentum_decay: float = 0.9
    align: str = "pooler_weighted"  # recommended and used in M-Attack-V2
    tm_idx: List[int] = field(
        default_factory=lambda: [-4, -3, -2, -1]
    )  # deprecated: kept for compatibility in older experiments
    beta: float = 0.3  # corresponds to lambda in the paper (pooler_weighted)
    use_retrieval: bool = False  # Flag to explicitly enable/disable retrieval
    multi_pass_num: int = 1  # corresponds to K in the paper (multi-pass count)


@dataclass
class ModelConfig:
    """Model-specific parameters"""

    input_res: int = 336
    crop_scale: tuple = (0.5, 0.9)
    ensemble: bool = True
    target_crop: bool = False
    device: str = "cuda:0"  # Can be "cpu", "cuda:0", "cuda:1", etc.
    backbone: list = (
        "L336",
        "B16",
        "B32",
        "Laion",
    )  # List of models to use: L336, B16, B32, Laion
    target_num: int = 1  # p + 1: one fixed target image + p retrieved targets


@dataclass
class MainConfig:
    """Main configuration combining all sub-configs"""

    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    blackbox: BlackboxConfig = field(default_factory=BlackboxConfig)
    attack: str = "pgd_multi_pass"  # main M-Attack-V2 attack
    generated_img_hash: Optional[str] = None  # optional eval-only hash override


# register config for different setting
@dataclass
class Ensemble3ModelsConfig(MainConfig):
    """Configuration for ensemble_3models.py"""

    data: DataConfig = field(default_factory=lambda: DataConfig(batch_size=1))
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(backbone=["B16", "B32", "Laion"])
    )
