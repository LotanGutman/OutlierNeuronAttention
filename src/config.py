from dataclasses import dataclass
from typing import Union, List, Tuple
import torch

@dataclass
class ModelConfig:
    d_model: int = 384
    num_heads: int = 6
    num_layers: int = 6
    r: Union[int, List[int], Tuple[int, ...]] = 8
    block_size: int = 1024
    tokenizer_name: str = "gpt2"
    seed: int = 42
    chunk_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_rope: bool = True
    mix_gate_bias_init: float = 0.0
    initializer_range: float = 0.02
    scale_residual_proj: bool = True
    
    # Ablation & Decay flags
    mha_heads_for_width_split: int = 0  # Width split baseline control
    fixed_blend_weight: bool = False  # Fixed blending weight ablation (no dynamic gating)
    pure_mha_low_rank_k: bool = False  # Reduced-QK MHA control: truncates both Q and K to d_QK=r with full d_h values (no GLA recurrence)
    attention_scale: float = 1.0  # Base attention scale factor, calibrated by YaRN
    rope_base: float = 10000.0
    learnable_alpha: bool = False  # Identity-initialized learnable channel-wise decay exponent (corresponds to \beta_h in the paper)

    @property
    def d_head(self) -> int:
        return self.d_model // self.num_heads


@dataclass
class InferenceConfig:
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    repetition_penalty: float = 1.15
    stream: bool = True
    seed: int = 42

