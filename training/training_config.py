from dataclasses import dataclass, field
import torch
from src.config import ModelConfig

@dataclass
class LanguageModelingExperimentConfig:
    # Model identity — change model_name to scale up/down (e.g. "100M")
    model_name: str = "30M"

    # Architecture
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            d_model=384,
            num_heads=6,
            num_layers=6,
            r=[32, 16, 16, 16, 16, 32],
            use_rope=True,
            block_size=1024
        )
    )

    # Dataset / caching
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    max_tokens: int = 1_000_000_000  # how many tokens to download & cache

    # Training hyperparameters
    batch_size: int = 32
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8  # 4 * 8 = 32
    seq_len: int = 1024
    vocab_size: int = 50257  # gpt2 vocab size
    train_steps: int = 53000
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 300
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print_every: int = 100
    save_every: int = 250
    grad_clip_norm: float = 1.0
    use_mixed_precision: bool = True

