from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    d_model: int = 384
    num_heads: int = 6
    num_layers: int = 6
    r: int = 8
    block_size: int = 1024
    tokenizer_name: str = "gpt2"
    refresh_steps: int = 100
    seed: int = 42
    chunk_size: int = 32 # should probably be 64 to match fla, need to remove j padding in chunk_gla_inlier so that this will fit in SRAM.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def d_head(self) -> int:
        return self.d_model // self.num_heads

@dataclass
class TrainingConfig:
    dataset_name: str = "roneneldan/TinyStories"
    tokenized_cache: str = "data/tokenized_data.pkl"
    train_cache: str = "data/train_tokens.pkl"
    val_cache: str = "data/val_tokens.pkl"
    val_split_ratio: float = 0.9
    max_tokens: int = 50_000_000
    batch_size: int = 2
    grad_accum_steps: int = 8
    max_iters: int = 5000
    eval_interval: int = 175
    save_interval: int = 250
    checkpoint_min_time: int = 600
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    eta_min: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "./checkpoints"
    compile_model: bool = False
    seed: int = 42
