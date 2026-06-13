"""Benchmark-specific configuration.

src/config.py is reserved purely for model and training configs.
"""

from dataclasses import dataclass, field
import torch
from src.config import ModelConfig, TrainingConfig

CACHE_PATH = "data/experiments_cache"

@dataclass
class PrefillExperimentConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_cfg: TrainingConfig = field(default_factory=TrainingConfig)
    seq_lengths: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
    cache_file_name: str = "profile_prefill_results.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass  
class RecallExperimentConfig:
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            d_model=256, num_heads=4, num_layers=2, r=8
        )
    )
    densities: tuple[int, ...] = (4, 8, 16)
    batch_size: int = 64
    seq_len: int = 64             # enough space for 16 pairs
    vocab_size: int = 128
    train_steps: int = 2000
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    seed: int = 42

    use_mixed_precision: bool = True
    grad_clip_norm: float = 1.0
    print_every: int = 1000
    num_eval_batches: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_file_name: str = "recall_results.pt"

@dataclass
class DecodeExperimentConfig:
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(d_model=2048, num_heads=16, num_layers=1, r=16)
    )
    warmup_steps: int = 10
    active_steps: int = 30
    seq_lengths: tuple[int, ...] = (512, 1024, 4096, 16384, 32768, 65536, 131072, 196608, 262144)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_file_name: str = "profile_decode_results.pt"
