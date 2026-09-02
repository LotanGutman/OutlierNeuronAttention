"""Benchmark-specific configuration.

src/config.py is reserved purely for model and training configs.
"""

from dataclasses import dataclass, field
import torch
from src.config import ModelConfig
from src.modules.benchmark_utils import AttentionType

CACHE_PATH = "data/experiments_cache"

@dataclass
class PrefillExperimentConfig:
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(d_model=2048, num_heads=16, num_layers=1, r=16)
    )
    seed: int = 42
    seq_lengths: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
    cache_file_name: str = "profile_prefill_results.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DecodeExperimentConfig:
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(d_model=2048, num_heads=16, num_layers=1, r=16)
    )
    warmup_steps: int = 10
    active_steps: int = 30
    seq_lengths: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_file_name: str = "profile_decode_results.pt"

@dataclass
class InductionExperimentConfig:
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            d_model=128,
            num_heads=4, 
            num_layers=4, 
            r=16,
            use_rope=True,
            mix_gate_bias_init=2.5,
            initializer_range=0.05,
            scale_residual_proj=False,
            block_size=2048
        )
    )
    batch_size: int = 32
    seq_len: int = 1024
    pattern_len: int = 8
    positional_jitter: int = 64
    vocab_size: int = 8192
    train_steps: int = 25000
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    
    disable_weight_decay_for_attention: bool = False # optional: Disables weight decay for W_q and W_k in attention layers. 
    
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print_every: int = 250
    grad_clip_norm: float = 1.0
    use_mixed_precision: bool = True
    
    seq_lengths: list = field(
        default_factory=lambda: [1024, 512, 256, 128, 64]
    )
    models_to_test: list = field(
        default_factory=lambda: [
            ("MHA", AttentionType.MHA, None),
            ("HOFA (r=8)", AttentionType.HOFA, 8),
            ("HOFA (r=10)", AttentionType.HOFA, 10),
            ("HOFA (r=16)", AttentionType.HOFA, 16),
            ("Gated DeltaNet", AttentionType.DELTA, None),
            ("GLA", AttentionType.GLA, None),
            ("Mamba", AttentionType.MAMBA, None)
        ]
    )

@dataclass
class InductionDegradationExperimentConfig(InductionExperimentConfig):
    batch_size: int = 32
    vocab_size: int = 8192
    seq_lengths: list = field(
        default_factory=lambda: [
            # Trail past cliff
            1200, 1160, 1120, 1080, 1040, 1000, 992, 984, 976, 968, 

            # Transition zone
            924, 920, 912, 904, 896, 892, 888, 884, 880, 872, 864, 856, 
            
            # Plateau pre cliff (P=1)
            848, 840, 832, 800, 768, 728, 688, 640
        ]
    )
    models_to_test: list = field(
        default_factory=lambda: [
            ("HOFA (r=8)", AttentionType.HOFA, 8)
        ]
    )
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            d_model=128,
            num_heads=4, 
            num_layers=4, 
            r=8,
            use_rope=True,
            mix_gate_bias_init=2.5,
            initializer_range=0.05,
            scale_residual_proj=False,
            block_size=2048
        )
    )

@dataclass
class CopyingExperimentConfig:
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            d_model=128,
            num_heads=4, 
            num_layers=4, 
            r=16,
            use_rope=True,
            mix_gate_bias_init=2.5,
            initializer_range=0.05,
            scale_residual_proj=False
        )
    )
    batch_size: int = 32
    pattern_len: int = 2
    gap_lengths: list = field(default_factory=lambda: [32, 64, 128, 256])
    seq_len: int = 1024 # Will be dynamically overridden per-gap
    vocab_size: int = 8192
    train_steps: int = 10000
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    
    # Optional - same as in the induction experiment.: Disables weight decay for W_q and W_k in attention layers. 
    disable_weight_decay_for_attention: bool = False
    
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print_every: int = 50
    grad_clip_norm: float = 1.0
    use_mixed_precision: bool = True
    models_to_test: list = field(
        default_factory=lambda: [
            ("MHA", AttentionType.MHA, None),
            ("HOFA (r=8)", AttentionType.HOFA, 8),
            ("HOFA (r=16)", AttentionType.HOFA, 16),
            ("Gated DeltaNet", AttentionType.DELTA, None),
            ("GLA", AttentionType.GLA, None),
            ("Mamba", AttentionType.MAMBA, None)
        ]
    )

@dataclass
class KEffExperimentConfig:
    num_sequences: int = 100
    seq_len: int = 1024
    batch_size: int = 1
    models_to_test: tuple[str, ...] = (
        "gpt2",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-2.8b",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.1-8B",
        "Qwen/Qwen2.5-7B"
    )
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    dataset_split: str = "train"
    cache_file_name: str = "layerwise_cumsum_results.pkl"

@dataclass
class DistanceExperimentConfig:
    num_sequences: int = 1000
    seq_len: int = 1024
    batch_size: int = 1
    models_to_test: tuple[str, ...] = (
        "350M_MHA",
        "350M_HOFA"
    )
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    dataset_split: str = "train"
    cache_file_name: str = "effective_distance_results.pkl"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class EvalExperimentConfig:
    """Configuration for zero-shot evaluations (HellaSwag, ARC, PIQA, WinoGrande, OpenBookQA)."""
    tasks: tuple[str, ...] = ("arc_easy", "arc_challenge", "piqa", "winogrande", "openbookqa", "hellaswag")
    limit: int | None = None
    seed: int = 42
    batch_size: int = 1  # Force to 1 to prevent padding corruption in unmasked HOFA
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    force_rerun: bool = False
    use_cache: bool = True
    cache_file_name: str = "zeroshot_eval_results.pt"
