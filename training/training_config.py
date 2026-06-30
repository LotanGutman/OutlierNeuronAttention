from dataclasses import dataclass, field
from typing import Union, List, Tuple
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


# ──────────────────────────────────────────────
# Pre-defined experiment configurations
# ──────────────────────────────────────────────

def _compute_steps(tokens: int, batch_size: int, seq_len: int) -> int:
    steps = tokens // (batch_size * seq_len)
    # Round to nearest hundred for cleanliness
    return (steps // 100) * 100


def _70M_base(**overrides) -> LanguageModelingExperimentConfig:
    """70M HOFA with bookend routing [32, 16, ..., 16, 32]."""
    params = dict(
        model_name="70M",
        model_config=ModelConfig(
            d_model=512,
            num_heads=8,
            num_layers=8,
            r=[32, 16, 16, 16, 16, 16, 16, 32],
            use_rope=True,
            block_size=1024,
        ),
        max_tokens=2_800_000_000,
        train_steps=_compute_steps(2_800_000_000, 32, 1024),
        learning_rate=6e-4,
        warmup_steps=500,
        print_every=200,
        save_every=500,
        micro_batch_size=4,
        gradient_accumulation_steps=8,
    )
    params.update(overrides)
    return LanguageModelingExperimentConfig(**params)


def make_70M_hofa() -> LanguageModelingExperimentConfig:
    return _70M_base()


def make_70M_pure_gla() -> LanguageModelingExperimentConfig:
    """Ablation: r=0 everywhere (pure GLA, no exact pathway)."""
    return _70M_base(model_name="70M_pure_GLA", model_config=ModelConfig(
        d_model=512, num_heads=8, num_layers=8, r=0,
        use_rope=True, block_size=1024,
    ))


def make_70M_r8() -> LanguageModelingExperimentConfig:
    """Ablation: uniform r=8 across all layers."""
    return _70M_base(model_name="70M_r8", model_config=ModelConfig(
        d_model=512, num_heads=8, num_layers=8, r=8,
        use_rope=True, block_size=1024,
    ))


def make_70M_r16() -> LanguageModelingExperimentConfig:
    """Ablation: uniform r=16 across all layers."""
    return _70M_base(model_name="70M_r16", model_config=ModelConfig(
        d_model=512, num_heads=8, num_layers=8, r=16,
        use_rope=True, block_size=1024,
    ))


def make_70M_r32() -> LanguageModelingExperimentConfig:
    """Ablation: uniform r=32 across all layers."""
    return _70M_base(model_name="70M_r32", model_config=ModelConfig(
        d_model=512, num_heads=8, num_layers=8, r=32,
        use_rope=True, block_size=1024,
    ))


def make_70M_pure_mha() -> LanguageModelingExperimentConfig:
    """Ablation: r=d_head=64 everywhere (pure MHA, no GLA)."""
    return _70M_base(model_name="70M_pure_MHA", model_config=ModelConfig(
        d_model=512, num_heads=8, num_layers=8, r=64,
        use_rope=True, block_size=1024,
    ))


def make_125M_hofa() -> LanguageModelingExperimentConfig:
    """125M HOFA with bookend routing [64, 32, 16, ..., 16, 32, 64]."""
    return LanguageModelingExperimentConfig(
        model_name="125M",
        model_config=ModelConfig(
            d_model=768,
            num_heads=12,
            num_layers=12,
            r=[64, 32, 16, 16, 16, 16, 16, 16, 16, 16, 32, 64],
            use_rope=True,
            block_size=1024,
        ),
        max_tokens=5_000_000_000,
        train_steps=_compute_steps(5_000_000_000, 32, 1024),
        learning_rate=5e-4,
        warmup_steps=750,
        print_every=200,
        save_every=500,
        micro_batch_size=2,
        gradient_accumulation_steps=16,  # 2 × 16 = 32
    )


def make_125M_mha() -> LanguageModelingExperimentConfig:
    """125M pure MHA baseline (r=d_head across all layers)."""
    return LanguageModelingExperimentConfig(
        model_name="125M_MHA",
        model_config=ModelConfig(
            d_model=768,
            num_heads=12,
            num_layers=12,
            r=64,  # r = d_head → full softmax
            use_rope=True,
            block_size=1024,
        ),
        max_tokens=5_000_000_000,
        train_steps=_compute_steps(5_000_000_000, 32, 1024),
        learning_rate=5e-4,
        warmup_steps=750,
        print_every=200,
        save_every=500,
        micro_batch_size=2,
        gradient_accumulation_steps=16,
    )


def make_350M_hofa() -> LanguageModelingExperimentConfig:
    """350M HOFA with bookend routing [64, 32, 16, ..., 16, 32, 64]."""
    return LanguageModelingExperimentConfig(
        model_name="350M",
        model_config=ModelConfig(
            d_model=1024,
            num_heads=16,
            num_layers=24,
            r=[64, 32] + [16] * 20 + [32, 64],
            use_rope=True,
            block_size=1024,
        ),
        max_tokens=14_000_000_000,
        train_steps=_compute_steps(14_000_000_000, 32, 1024),
        learning_rate=3e-4,
        warmup_steps=2000,
        print_every=500,
        save_every=1000,
        micro_batch_size=1,
        gradient_accumulation_steps=32,  # 1 × 32 = 32
    )


def make_350M_mha() -> LanguageModelingExperimentConfig:
    """350M pure MHA baseline (r=d_head across all layers)."""
    return LanguageModelingExperimentConfig(
        model_name="350M_MHA",
        model_config=ModelConfig(
            d_model=1024,
            num_heads=16,
            num_layers=24,
            r=64,
            use_rope=True,
            block_size=1024,
        ),
        max_tokens=14_000_000_000,
        train_steps=_compute_steps(14_000_000_000, 32, 1024),
        learning_rate=3e-4,
        warmup_steps=2000,
        print_every=500,
        save_every=1000,
        micro_batch_size=1,
        gradient_accumulation_steps=32,
    )

