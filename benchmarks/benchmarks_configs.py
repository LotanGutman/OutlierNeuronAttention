"""Benchmark-specific configuration.

src/config.py is reserved purely for model and training configs.
"""

from dataclasses import dataclass, field
import torch
from src.config import ModelConfig

@dataclass  
class RecallExperimentConfig:
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            d_model=128, num_heads=2, num_layers=2, r=8, m=32, m_O=64
        )
    )
    densities: tuple[int, ...] = (4, 8, 16)
    batch_size: int = 64
    seq_len: int = 64
    vocab_size: int = 128
    train_steps: int = 10000
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    seed: int = 42
    
    use_mixed_precision: bool = True
    grad_clip_norm: float = 1.0
    print_every: int = 500
    num_eval_batches: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"