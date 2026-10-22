from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional, Callable
from enum import Enum
import torch
from src.config import ModelConfig

class DatasetType(str, Enum):
    FINEWEB = "fineweb"
    PG19 = "pg19"


"""
important - 
the saved dataset cache file will be named accordingly to the model_name before the first "_".
Therefore it should be set to "30M_{name}" or "...M_{name}" to avoid overwriting the cache file
when training different models architectures of the same size (require same data)
"""

@dataclass
class LanguageModelingExperimentConfig:
    # Model identity — change model_name to scale up/down
    model_name: str = "30M_HOFA"
    plot_name: str = "30M HOFA"

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
    dataset_name: DatasetType = DatasetType.FINEWEB
    max_tokens: int = 1_000_000_000  # how many tokens to download & cache
    train_cache_path: Optional[str] = None
    val_cache_path: Optional[str] = None

    # Training hyperparameters
    batch_size: int = 32
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 16  # 2 * 16 = 32
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

    val_every: int = 500
    val_num_batches: int = 50
    activation_checkpointing_interval: Optional[int] = None
    gate_lr_multiplier: Optional[float] = None

@dataclass
class YaRNConfig:
    model: LanguageModelingExperimentConfig
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    original_max_seq_len: Optional[int] = None
    base_checkpoint_path: Optional[str] = None

    def __post_init__(self):
        if self.base_checkpoint_path is None:
            base_name = self.model.model_name.replace("_cpt", "")
            self.base_checkpoint_path = f"data/training/{base_name}/checkpoint_best_val.pt"


# ──────────────────────────────────────────────
# Pre-defined experiment configurations
# ──────────────────────────────────────────────

def _compute_steps(tokens: int, batch_size: int, seq_len: int) -> int:
    steps = tokens / (batch_size * seq_len)
    # round to nearest 100 (so that it evaluates last point)
    return int(round(steps / 100.0)) * 100

# 13M models (For quick debugging)
def make_13M_HOFA() -> LanguageModelingExperimentConfig:
    """13M HOFA for rapid testing. d_head=64."""
    return LanguageModelingExperimentConfig(
        model_name="13M_HOFA",
        plot_name="13M HOFA",
        model_config=ModelConfig(
            d_model=256, num_heads=4, num_layers=4,
            r=[32, 16, 16, 32],
            use_rope=True, block_size=1024,
        ),
        max_tokens=200_000_000,
        train_steps=_compute_steps(200_000_000, 32, 1024),
        learning_rate=1e-3, warmup_steps=100, print_every=50, save_every=200,
        micro_batch_size=8,
        gradient_accumulation_steps=4,
    )


def make_13M_MHA() -> LanguageModelingExperimentConfig:
    """13M pure MHA baseline. r=d_head=64."""
    return LanguageModelingExperimentConfig(
        model_name="13M_MHA",
        plot_name="13M MHA",
        model_config=ModelConfig(
            d_model=256, num_heads=4, num_layers=4,
            r=64,
            use_rope=True, block_size=1024,
        ),
        max_tokens=200_000_000,
        train_steps=1000,
        learning_rate=1e-3, warmup_steps=100, print_every=50, save_every=200,
        micro_batch_size=8,
        gradient_accumulation_steps=4,
    )


# 30M models
def make_30M_HOFA() -> LanguageModelingExperimentConfig:
    """30M HOFA base model."""
    return LanguageModelingExperimentConfig(
        model_name="30M_HOFA",
        plot_name="30M HOFA",
        model_config=ModelConfig(
            d_model=384, num_heads=6, num_layers=6,
            r=[32, 16, 16, 16, 16, 32],
            use_rope=True, block_size=1024,
        ),
        max_tokens=1_000_000_000,
        train_steps=_compute_steps(1_000_000_000, 32, 1024),
        learning_rate=6e-4, warmup_steps=300, print_every=100, save_every=250,
        micro_batch_size=2,
        gradient_accumulation_steps=16,
    )


def make_30M_MHA() -> LanguageModelingExperimentConfig:
    """30M pure MHA baseline. r=d_head=64."""
    return LanguageModelingExperimentConfig(
        model_name="30M_MHA",
        plot_name="30M MHA",
        model_config=ModelConfig(
            d_model=384, num_heads=6, num_layers=6,
            r=64,
            use_rope=True, block_size=1024,
        ),
        max_tokens=1_000_000_000,
        train_steps=_compute_steps(1_000_000_000, 32, 1024),
        learning_rate=6e-4, warmup_steps=300, print_every=100, save_every=250,
        micro_batch_size=2,
        gradient_accumulation_steps=16,
    )


# 70M models (Ablations)
def _70M_base(**overrides) -> LanguageModelingExperimentConfig:
    """70M base."""
    params = dict(
        model_name="70M",
        plot_name="70M",
        model_config=ModelConfig(
            d_model=512, num_heads=8, num_layers=8, r=16,
            use_rope=True, block_size=1024,
        ),
        max_tokens=2_800_000_000,
        train_steps=_compute_steps(2_800_000_000, 32, 1024),
        learning_rate=6e-4, warmup_steps=500, print_every=200, save_every=500,
        micro_batch_size=16,
        gradient_accumulation_steps=2,
    )
    params.update(overrides)
    return LanguageModelingExperimentConfig(**params)

def make_70M_MHA() -> LanguageModelingExperimentConfig:
    return _70M_base(model_name="70M_MHA", plot_name="MHA", model_config=ModelConfig(
        d_model=512, num_heads=8, num_layers=8, r=64,
        use_rope=True, block_size=1024,
    ))

def make_70M_HOFA() -> LanguageModelingExperimentConfig:
    return _70M_base(model_name="70M_HOFA", plot_name="U-Shaped HOFA", model_config=ModelConfig(
        d_model=512, num_heads=8, num_layers=8, r=[64, 32, 16, 16, 16, 16, 32, 64],
        use_rope=True, block_size=1024,
    ))

def make_70M_HOFA_flat32() -> LanguageModelingExperimentConfig:
    return _70M_base(model_name="70M_HOFA_flat32", plot_name="HOFA (Flat r=32)", model_config=ModelConfig(
        d_model=512, num_heads=8, num_layers=8, r=32,
        use_rope=True, block_size=1024,
    ))

def make_70M_HOFA_depth_axis() -> LanguageModelingExperimentConfig:
    return _70M_base(model_name="70M_HOFA_depth_axis", plot_name="Depth-Axis Hybrid", model_config=ModelConfig(
        d_model=512, num_heads=8, num_layers=8, r=[0, 64, 64, 0, 0, 64, 64, 0],
        use_rope=True, block_size=1024,
    ))

def make_70M_HOFA_width_axis() -> LanguageModelingExperimentConfig:
    return _70M_base(model_name="70M_HOFA_width_axis", plot_name="Width-Axis Hybrid", model_config=ModelConfig(
        d_model=512, num_heads=8, num_layers=8, r=0,
        mha_heads_for_width_split=4,
        use_rope=True, block_size=1024,
    ))

def make_70M_HOFA_fixed_blend() -> LanguageModelingExperimentConfig:
    return _70M_base(model_name="70M_HOFA_fixed_blend", plot_name="No Mixing Gate HOFA", model_config=ModelConfig(
        d_model=512, num_heads=8, num_layers=8, r=[64, 32, 16, 16, 16, 16, 32, 64],
        fixed_blend_weight=True,
        use_rope=True, block_size=1024,
    ))


# 125M models (Main)
def make_125M_HOFA() -> LanguageModelingExperimentConfig:
    """125M HOFA. d_head=128 (768/6)."""
    return LanguageModelingExperimentConfig(
        model_name="125M_HOFA",
        plot_name="125M HOFA",
        model_config=ModelConfig(
            d_model=768, num_heads=6, num_layers=12,
            r=[64, 32, 16, 16, 16, 16, 16, 16, 16, 16, 32, 64],
            use_rope=True, block_size=1024,
        ),
        max_tokens=5_000_000_000,
        train_steps=_compute_steps(5_000_000_000, 32, 1024),
        learning_rate=5e-4, warmup_steps=750, print_every=200, save_every=500,
        micro_batch_size=16,
        gradient_accumulation_steps=2,
    )


def make_125M_MHA() -> LanguageModelingExperimentConfig:
    """125M pure MHA baseline. r=d_head=128."""
    return LanguageModelingExperimentConfig(
        model_name="125M_MHA",
        plot_name="125M MHA",
        model_config=ModelConfig(
            d_model=768, num_heads=6, num_layers=12, r=128,
            use_rope=True, block_size=1024,
        ),
        max_tokens=5_000_000_000,
        train_steps=_compute_steps(5_000_000_000, 32, 1024),
        learning_rate=5e-4, warmup_steps=750, print_every=200, save_every=500,
        micro_batch_size=16,
        gradient_accumulation_steps=2,
    )

# 350M models (Main)
def make_350M_HOFA() -> LanguageModelingExperimentConfig:
    """350M HOFA. d_head=128 (1024/8)."""
    return LanguageModelingExperimentConfig(
        model_name="350M_HOFA",
        plot_name="350M HOFA",
        model_config=ModelConfig(
            d_model=1024, num_heads=8, num_layers=24,
            r=[64, 32, 32] + [16] * 18 + [32, 32, 64],
            use_rope=True, block_size=1024,
        ),
        max_tokens=14_000_000_000,
        train_steps=_compute_steps(14_000_000_000, 32, 1024),
        learning_rate=3e-4, warmup_steps=2000, print_every=500, save_every=1000,
        micro_batch_size=16,
        gradient_accumulation_steps=2,
    )


def make_350M_MHA() -> LanguageModelingExperimentConfig:
    """350M pure MHA baseline. r=d_head=128."""
    return LanguageModelingExperimentConfig(
        model_name="350M_MHA",
        plot_name="350M MHA",
        model_config=ModelConfig(
            d_model=1024, num_heads=8, num_layers=24, r=128,
            use_rope=True, block_size=1024,
        ),
        max_tokens=14_000_000_000,
        train_steps=_compute_steps(14_000_000_000, 32, 1024),
        learning_rate=3e-4, warmup_steps=2000, print_every=500, save_every=1000,
        micro_batch_size=8,
        gradient_accumulation_steps=4,
    )







# Continual pretraining (CPT) configs

def make_13M_HOFA_cpt() -> YaRNConfig:
    cfg = make_13M_HOFA()
    cfg.model_name = "13M_HOFA_cpt"
    cfg.plot_name = "13M HOFA (16k CPT)"
    cfg.seq_len = 16384
    cfg.model_config.block_size = 16384
    cfg.max_tokens = 100_000_000
    cfg.batch_size = 16
    cfg.micro_batch_size = 1
    cfg.gradient_accumulation_steps = 16
    cfg.train_steps = _compute_steps(100_000_000, 16, 16384)
    cfg.learning_rate = 5e-5
    cfg.gate_lr_multiplier = 10.0
    cfg.weight_decay = 0.0
    cfg.warmup_steps = 30
    cfg.val_every = 50
    cfg.val_num_batches = 16
    cfg.save_every = 50
    cfg.print_every = 10
    cfg.dataset_name = DatasetType.PG19
    cfg.train_cache_path = "data/datasets/pg19_16k_train.bin"
    cfg.val_cache_path = "data/datasets/pg19_16k_val.bin"
    return YaRNConfig(model=cfg)


def make_13M_MHA_cpt() -> YaRNConfig:
    cfg = make_13M_MHA()
    cfg.model_name = "13M_MHA_cpt"
    cfg.plot_name = "13M MHA (16k CPT)"
    cfg.seq_len = 16384
    cfg.model_config.block_size = 16384
    cfg.max_tokens = 100_000_000
    cfg.batch_size = 16
    cfg.micro_batch_size = 1
    cfg.gradient_accumulation_steps = 16
    cfg.train_steps = _compute_steps(100_000_000, 16, 16384)
    cfg.learning_rate = 5e-5
    cfg.weight_decay = 0.0
    cfg.warmup_steps = 30
    cfg.val_every = 50
    cfg.val_num_batches = 16
    cfg.save_every = 50
    cfg.print_every = 10
    cfg.dataset_name = DatasetType.PG19
    cfg.train_cache_path = "data/datasets/pg19_16k_train.bin"
    cfg.val_cache_path = "data/datasets/pg19_16k_val.bin"
    return YaRNConfig(model=cfg)


def make_30M_HOFA_cpt() -> YaRNConfig:
    cfg = make_30M_HOFA()
    cfg.model_name = "30M_HOFA_cpt"
    cfg.plot_name = "30M HOFA (16k CPT)"
    cfg.seq_len = 16384
    cfg.model_config.block_size = 16384
    cfg.max_tokens = 100_000_000
    cfg.batch_size = 16
    cfg.micro_batch_size = 1
    cfg.gradient_accumulation_steps = 16
    cfg.train_steps = _compute_steps(100_000_000, 16, 16384)
    cfg.learning_rate = 3e-5
    cfg.gate_lr_multiplier = 10.0
    cfg.weight_decay = 0.0
    cfg.warmup_steps = 30
    cfg.val_every = 50
    cfg.val_num_batches = 16
    cfg.save_every = 50
    cfg.print_every = 10
    cfg.dataset_name = DatasetType.PG19
    cfg.train_cache_path = "data/datasets/pg19_16k_train.bin"
    cfg.val_cache_path = "data/datasets/pg19_16k_val.bin"
    return YaRNConfig(model=cfg)


def make_30M_MHA_cpt() -> YaRNConfig:
    cfg = make_30M_MHA()
    cfg.model_name = "30M_MHA_cpt"
    cfg.plot_name = "30M MHA (16k CPT)"
    cfg.seq_len = 16384
    cfg.model_config.block_size = 16384
    cfg.max_tokens = 100_000_000
    cfg.batch_size = 16
    cfg.micro_batch_size = 1
    cfg.gradient_accumulation_steps = 16
    cfg.train_steps = _compute_steps(100_000_000, 16, 16384)
    cfg.learning_rate = 3e-5
    cfg.weight_decay = 0.0
    cfg.warmup_steps = 30
    cfg.val_every = 50
    cfg.val_num_batches = 16
    cfg.save_every = 50
    cfg.print_every = 10
    cfg.dataset_name = DatasetType.PG19
    cfg.train_cache_path = "data/datasets/pg19_16k_train.bin"
    cfg.val_cache_path = "data/datasets/pg19_16k_val.bin"
    return YaRNConfig(model=cfg)


def make_350M_HOFA_cpt() -> YaRNConfig:
    cfg = make_350M_HOFA()
    base_seq_len = cfg.seq_len
    cfg.model_name = "350M_HOFA_cpt"
    cfg.plot_name = "350M HOFA (16k CPT)"
    cfg.seq_len = 16384
    cfg.model_config.block_size = 16384
    cfg.max_tokens = 100_000_000
    cfg.batch_size = 16
    cfg.micro_batch_size = 1
    cfg.gradient_accumulation_steps = 16
    cfg.train_steps = _compute_steps(100_000_000, 16, 16384)
    cfg.learning_rate = 2e-5
    cfg.gate_lr_multiplier = 10.0
    cfg.weight_decay = 0.0
    cfg.warmup_steps = 20
    cfg.val_every = 50
    cfg.val_num_batches = 8
    cfg.save_every = 50
    cfg.print_every = 10
    cfg.dataset_name = DatasetType.PG19
    cfg.train_cache_path = "data/datasets/pg19_16k_train.bin"
    cfg.val_cache_path = "data/datasets/pg19_16k_val.bin"
    cfg.activation_checkpointing_interval = 2
    return YaRNConfig(model=cfg, original_max_seq_len=base_seq_len)


def make_350M_MHA_cpt() -> YaRNConfig:
    cfg = make_350M_MHA()
    base_seq_len = cfg.seq_len
    cfg.model_name = "350M_MHA_cpt"
    cfg.plot_name = "350M MHA (16k CPT)"
    cfg.seq_len = 16384
    cfg.model_config.block_size = 16384
    cfg.max_tokens = 100_000_000
    cfg.batch_size = 16
    cfg.micro_batch_size = 1
    cfg.gradient_accumulation_steps = 16
    cfg.train_steps = _compute_steps(100_000_000, 16, 16384)
    cfg.learning_rate = 2e-5
    cfg.weight_decay = 0.0
    cfg.warmup_steps = 20
    cfg.val_every = 50
    cfg.val_num_batches = 8
    cfg.save_every = 50
    cfg.print_every = 10
    cfg.dataset_name = DatasetType.PG19
    cfg.train_cache_path = "data/datasets/pg19_16k_train.bin"
    cfg.val_cache_path = "data/datasets/pg19_16k_val.bin"
    cfg.activation_checkpointing_interval = 2
    return YaRNConfig(model=cfg, original_max_seq_len=base_seq_len)


MODEL_CONFIG_REGISTRY = {
    # Pretraining configs
    ("13M", "HOFA", False): make_13M_HOFA,
    ("13M", "MHA", False): make_13M_MHA,
    ("30M", "HOFA", False): make_30M_HOFA,
    ("30M", "MHA", False): make_30M_MHA,
    ("70M", "HOFA", False): make_70M_HOFA,
    ("70M", "MHA", False): make_70M_MHA,
    ("125M", "HOFA", False): make_125M_HOFA,
    ("125M", "MHA", False): make_125M_MHA,
    ("350M", "HOFA", False): make_350M_HOFA,
    ("350M", "MHA", False): make_350M_MHA,

    # Continual Pretraining (YaRN) configs
    ("13M", "HOFA", True): make_13M_HOFA_cpt,
    ("13M", "MHA", True): make_13M_MHA_cpt,
    ("30M", "HOFA", True): make_30M_HOFA_cpt,
    ("30M", "MHA", True): make_30M_MHA_cpt,
    ("350M", "HOFA", True): make_350M_HOFA_cpt,
    ("350M", "MHA", True): make_350M_MHA_cpt,

    # Direct string name lookups
    "13M_HOFA": make_13M_HOFA,
    "13M_MHA": make_13M_MHA,
    "30M_HOFA": make_30M_HOFA,
    "30M_MHA": make_30M_MHA,
    "70M_HOFA": make_70M_HOFA,
    "70M_MHA": make_70M_MHA,
    "125M_HOFA": make_125M_HOFA,
    "125M_MHA": make_125M_MHA,
    "350M_HOFA": make_350M_HOFA,
    "350M_MHA": make_350M_MHA,

    "13M_HOFA_CPT": make_13M_HOFA_cpt,
    "13M_MHA_CPT": make_13M_MHA_cpt,
    "30M_HOFA_CPT": make_30M_HOFA_cpt,
    "30M_MHA_CPT": make_30M_MHA_cpt,
    "350M_HOFA_CPT": make_350M_HOFA_cpt,
    "350M_MHA_CPT": make_350M_MHA_cpt,
}


def get_model_config(
    scale: str = "350M",
    model: Optional[str] = "hofa",
    is_cpt: bool = False
) -> Union[LanguageModelingExperimentConfig, YaRNConfig]:
    """Retrieve model experiment configuration by (scale, model, is_cpt) tuple or composite name."""
    clean_str = scale.strip().upper()

    # Check if a composite string name was provided (e.g. '125M_MHA' or '13M_HOFA_CPT')
    if "_" in clean_str or model is None:
        if clean_str in MODEL_CONFIG_REGISTRY:
            return MODEL_CONFIG_REGISTRY[clean_str]()
        raise NotImplementedError(
            f"No config registered for '{scale}'. "
            f"Available named configs: {[k for k in MODEL_CONFIG_REGISTRY if isinstance(k, str)]}"
        )

    # Otherwise lookup by (scale, model, is_cpt)
    key = (clean_str, model.strip().upper(), bool(is_cpt))
    if key in MODEL_CONFIG_REGISTRY:
        return MODEL_CONFIG_REGISTRY[key]()

    available_tuples = [k for k in MODEL_CONFIG_REGISTRY if isinstance(k, tuple)]
    raise NotImplementedError(
        f"No config for combination: scale='{scale}', model='{model}', is_cpt={is_cpt}. "
        f"Available combinations: {available_tuples}"
    )



