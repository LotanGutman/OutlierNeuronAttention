"""Benchmark-specific configuration.

src/config.py is reserved purely for model and training configs.
"""

from dataclasses import dataclass, field
import torch
from src.config import ModelConfig, TrainingConfig

CACHE_PATH = "data/experiments_cache"

@dataclass
class ProfileExperimentConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_cfg: TrainingConfig = field(default_factory=TrainingConfig)
    cache_file_name: str = "profiling_results.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass  
class RecallExperimentConfig:
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            d_model=256, num_heads=4, num_layers=4, r=8, refresh_steps=999999
        )
    )
    densities: tuple[int, ...] = (8, ) # (4, 8, 16)
    batch_size: int = 64
    seq_len: int = 64
    vocab_size: int = 128
    train_steps: int = 20000
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    seed: int = 42
    
    use_mixed_precision: bool = True
    grad_clip_norm: float = 1.0
    print_every: int = 500
    num_eval_batches: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_file_name: str = "recall_results.pt"
