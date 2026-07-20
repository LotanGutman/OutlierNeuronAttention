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

