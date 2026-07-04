"""Benchmark-specific configuration.

src/config.py is reserved purely for model and training configs.
"""

from dataclasses import dataclass, field
import torch
from src.config import ModelConfig

CACHE_PATH = "data/experiments_cache"

@dataclass
class PrefillExperimentConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 42
    seq_lengths: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
    cache_file_name: str = "profile_prefill_results.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"



@dataclass
class InductionExperimentConfig:
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            d_model=128,
            num_heads=4, 
            num_layers=4, 
            r=16,
            use_rope=True
        )
    )
    batch_size: int = 32
    seq_len: int = 1024
    vocab_size: int = 8192
    train_steps: int = 20000
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print_every: int = 250
    grad_clip_norm: float = 1.0
    use_mixed_precision: bool = True

@dataclass
class DecodeExperimentConfig:
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(d_model=2048, num_heads=16, num_layers=1, r=16)
    )
    warmup_steps: int = 10
    active_steps: int = 30
    seq_lengths: tuple[int, ...] = (512, 1024, 4096, 16384, 32768, 65536, 131072, 196608)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_file_name: str = "profile_decode_results.pt"

@dataclass
class LongDistancePPLConfig:
    """Long-distance perplexity evaluation (Child et al. 2019)."""
    batch_size: int = 16
    seq_len: int = 1024
    distance_threshold: int = 512
    rare_token_threshold: int = 500
    num_batches: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_file_name: str = "long_distance_ppl_results.pt"

@dataclass
class DistanceMetricConfig:
    """Effective attention distance evaluation (Child et al. 2019)."""
    model_name: str = "30M"
    batch_size: int = 4
    seq_len: int = 1024
    num_sequences: int = 1024
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_file_name: str = "distance_metric_results.pt"

@dataclass
class KEffExperimentConfig:
    num_sequences: int = 100
    seq_len: int = 1024
    batch_size: int = 2
    models_to_test: tuple[str, ...] = (
        "gpt2",
        "EleutherAI/pythia-410m",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B"
    )
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    dataset_split: str = "train"
    cache_file_name: str = "layerwise_cumsum_results.pkl"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class EvalExperimentConfig:
    """Configuration for zero-shot evaluations (HellaSwag, LAMBADA, etc.)."""
    limit: int | None = 1000
    seed: int = 42
    batch_size: str | int = "auto"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
