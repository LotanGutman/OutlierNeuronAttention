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
    refresh_steps: int = 100
    seed: int = 42
    chunk_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_rope: bool = True # make it true by default
    mix_gate_bias_init: float = 0.0
    initializer_range: float = 0.02 # model weight initialization std    
    scale_residual_proj: bool = True # Whether to scale down the residual projection (out_proj) initialization by 1/sqrt(2 * num_layers).
    
    # Ablation flags
    mha_heads_for_width_split: int = 0 # For Hymba
    fixed_blend_weight: bool = False # No gating ablation
    
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

