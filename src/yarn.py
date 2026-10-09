import os
import math
import torch
import torch.nn as nn
from typing import Optional
import gc

from training.training_config import YaRNConfig
from src.HybridOutlierFactorizedAttentionTrain import SubwordLM as TrainSubwordLM
from src.HybridOutlierFactorizedAttention import SubwordLM as InferenceSubwordLM


def yarn_find_correction_dim(beta: float, dim: int, base: float, max_seq_len: float) -> float:
    """Computes the YaRN correction dimension boundary: d(beta) = (dim * ln(L / (2*pi*beta))) / (2 * ln(base))."""
    return (dim * math.log(max_seq_len / (2.0 * math.pi * beta))) / (2.0 * math.log(base))


class YaRNModel(nn.Module):
    """Contained YaRN context extension class for MHA \ HOFA"""
    def __init__(
        self,
        yarn_config: YaRNConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        load_checkpoint: bool = True,
        checkpoint_path: Optional[str] = None,
        inference_model: bool = False,
    ):
        super().__init__()
        self.yarn_config = yarn_config
        self.device = device
        self.dtype = dtype
        self.inference_model = inference_model
        self.original_max_seq_len = (
            yarn_config.original_max_seq_len
            if yarn_config.original_max_seq_len is not None
            else yarn_config.model.model_config.block_size
        )

        if self.inference_model:
            self.model = InferenceSubwordLM(
                yarn_config.model.vocab_size,
                yarn_config.model.model_config,
            ).to(device=device, dtype=dtype)
        else:
            self.model = TrainSubwordLM(
                yarn_config.model.vocab_size,
                yarn_config.model.model_config,
            ).to(device=device, dtype=dtype)

        if load_checkpoint:
            path = checkpoint_path or f"data/training/{yarn_config.model.model_name}/checkpoint_best_val.pt"
            if os.path.exists(path):
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                if isinstance(ckpt, dict):
                    ckpt.pop("optimizer_state_dict", None)
                    ckpt.pop("optimizer", None)
                self.model.load_state_dict(state_dict)
                del ckpt, state_dict
                gc.collect()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def forward_step(self, *args, **kwargs):
        if self.inference_model:
            return self.model.forward_step(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def set_context_length(self, seq_len: int) -> int:
        scale = max(1.0, float(seq_len) / float(self.original_max_seq_len))
        patched_count = 0

        for layer in self.model.layers:
            attn = layer.attn
            if hasattr(attn, "rotary_emb") and attn.rotary_emb is not None:
                rot = attn.rotary_emb
                dim = rot.dim
                dev = rot.inv_freq.device
                base = float(self.yarn_config.model.model_config.rope_base)

                # Base RoPE inverse frequencies
                idx = torch.arange(0, dim, 2, dtype=torch.float32, device=dev)
                inv_freq_extrapolation = 1.0 / (base ** (idx / dim))

                if scale > 1.0:
                    L = float(self.original_max_seq_len)
                    n_freqs = dim // 2

                    beta_fast = float(self.yarn_config.beta_fast)  # default 32.0
                    beta_slow = float(self.yarn_config.beta_slow)  # default 1.0

                    low = max(0, math.floor(yarn_find_correction_dim(beta_fast, dim, base, L)))
                    high = min(n_freqs - 1, math.ceil(yarn_find_correction_dim(beta_slow, dim, base, L)))

                    # Linear ramp across rotary dimensions
                    freq_idx = torch.arange(n_freqs, dtype=torch.float32, device=dev)
                    if high <= low:
                        high = low + 1e-3
                    ramp = ((freq_idx - low) / (high - low)).clamp(0.0, 1.0)

                    # High-frequency dimensions extrapolate; low-frequency dimensions interpolate
                    inv_freq_interpolation = inv_freq_extrapolation / scale
                    inv_freq = (
                        ramp * inv_freq_interpolation
                        + (1.0 - ramp) * inv_freq_extrapolation
                    )

                    # Standard YaRN magnitude scaling: mscale = 1.0 + 0.1 * ln(scale)
                    mscale = 1.0 + 0.1 * math.log(scale)
                else:
                    inv_freq = inv_freq_extrapolation
                    mscale = 1.0

                rot.inv_freq = inv_freq
                cache_len = max(rot.max_seq_len_cached, seq_len)
                t = torch.arange(cache_len, dtype=torch.float32, device=dev)
                freqs = torch.outer(t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)

                rot.register_buffer("cos_cached", (emb.cos() * mscale).to(self.dtype), persistent=False)
                rot.register_buffer("sin_cached", (emb.sin() * mscale).to(self.dtype), persistent=False)
                rot.max_seq_len_cached = cache_len
                patched_count += 1

        return patched_count

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
