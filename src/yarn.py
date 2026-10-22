"""
Deciding on what the original base seq length is a bit messy, so working that out is a big TODO ;)
"""

import os
import math
import torch
import torch.nn as nn
from typing import Optional
import gc

from training.training_config import YaRNConfig, get_model_config
from src.HybridOutlierFactorizedAttentionTrain import SubwordLM as TrainSubwordLM
from src.HybridOutlierFactorizedAttention import SubwordLM as InferenceSubwordLM


def yarn_find_correction_dim(beta: float, dim: int, base: float, max_seq_len: float) -> float:
    """Computes the YaRN correction dimension boundary: d(beta) = (dim * ln(L / (2*pi*beta))) / (2 * ln(base))."""
    return (dim * math.log(max_seq_len / (2.0 * math.pi * beta))) / (2.0 * math.log(base))


def patch_model_yarn(
    model: nn.Module,
    target_seq_len: int,
    original_seq_len: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    dtype: torch.dtype = torch.bfloat16,
    scale: Optional[float] = None,
) -> int:
    if scale is None:
        scale = max(1.0, float(target_seq_len) / float(original_seq_len))
    patched = 0

    rope_base = float(getattr(getattr(model, "config", None), "rope_base", 10000.0))

    layers = getattr(model, "layers", [])
    for layer in layers:
        attn = getattr(layer, "attn", None)
        if attn is not None and hasattr(attn, "rotary_emb") and attn.rotary_emb is not None:
            rot = attn.rotary_emb
            dim = rot.dim
            dev = rot.inv_freq.device

            idx = torch.arange(0, dim, 2, dtype=torch.float32, device=dev)
            inv_freq_extrapolation = 1.0 / (rope_base ** (idx / dim))

            if scale > 1.0:
                L = float(original_seq_len)
                n_freqs = dim // 2
                low = max(0, math.floor(yarn_find_correction_dim(beta_fast, dim, rope_base, L)))
                high = min(n_freqs - 1, math.ceil(yarn_find_correction_dim(beta_slow, dim, rope_base, L)))

                freq_idx = torch.arange(n_freqs, dtype=torch.float32, device=dev)
                if high <= low:
                    high = low + 1e-3
                ramp = ((freq_idx - low) / (high - low)).clamp(0.0, 1.0)
                inv_freq_interpolation = inv_freq_extrapolation / scale
                inv_freq = ramp * inv_freq_interpolation + (1.0 - ramp) * inv_freq_extrapolation
                effective_mscale = 1.0 + 0.1 * math.log(scale)
            else:
                inv_freq = inv_freq_extrapolation
                effective_mscale = 1.0

            rot.inv_freq = inv_freq
            cache_len = max(rot.max_seq_len_cached, target_seq_len)
            t = torch.arange(cache_len, dtype=torch.float32, device=dev)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

            rot.register_buffer("cos_cached", (emb.cos() * effective_mscale).to(dtype), persistent=False)
            rot.register_buffer("sin_cached", (emb.sin() * effective_mscale).to(dtype), persistent=False)
            rot.max_seq_len_cached = cache_len
            patched += 1

    return patched


def swap_rotary_to_native(model: nn.Module, native_seq_len: int = 1024, dtype: torch.dtype = torch.bfloat16) -> list:
    """Saves YaRN rotary buffers and installs native (s=1, unscaled) tables."""
    saved_states = []
    rope_base = float(getattr(getattr(model, "config", None), "rope_base", 10000.0))

    layers = getattr(model, "layers", [])
    for i, layer in enumerate(layers):
        attn = getattr(layer, "attn", None)
        if attn is not None and hasattr(attn, "rotary_emb") and attn.rotary_emb is not None:
            rot = attn.rotary_emb
            dev = rot.inv_freq.device
            dim = rot.dim

            saved_states.append((
                i,
                rot.inv_freq.clone(),
                rot.cos_cached.clone(),
                rot.sin_cached.clone(),
                rot.max_seq_len_cached,
            ))

            idx = torch.arange(0, dim, 2, dtype=torch.float32, device=dev)
            inv_freq_native = 1.0 / (rope_base ** (idx / dim))
            cache_len = max(rot.max_seq_len_cached, native_seq_len)
            t = torch.arange(cache_len, dtype=torch.float32, device=dev)
            freqs = torch.outer(t, inv_freq_native)
            emb = torch.cat((freqs, freqs), dim=-1)

            rot.inv_freq = inv_freq_native
            rot.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
            rot.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
            rot.max_seq_len_cached = cache_len

    return saved_states


def restore_rotary_from_saved(model: nn.Module, saved_states: list) -> None:
    """Restores saved YaRN rotary buffers and asserts bitwise match."""
    layers = getattr(model, "layers", [])
    for item in saved_states:
        i, inv_freq_orig, cos_orig, sin_orig, max_cached = item
        rot = layers[i].attn.rotary_emb
        rot.inv_freq = inv_freq_orig
        rot.register_buffer("cos_cached", cos_orig, persistent=False)
        rot.register_buffer("sin_cached", sin_orig, persistent=False)
        rot.max_seq_len_cached = max_cached
        assert torch.equal(rot.inv_freq, inv_freq_orig)
        assert torch.equal(rot.cos_cached, cos_orig)
        assert torch.equal(rot.sin_cached, sin_orig)


class YaRNModel(nn.Module):
    """YaRN context extension class for HOFA"""
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
        if yarn_config.original_max_seq_len is not None:
            self.original_max_seq_len = yarn_config.original_max_seq_len
        elif "_cpt" in getattr(yarn_config.model, "model_name", "").lower():
            base_model_name = yarn_config.model.model_name.replace("_cpt", "").replace("_CPT", "")
            try:
                self.original_max_seq_len = get_model_config(base_model_name).seq_len
            except Exception:
                self.original_max_seq_len = getattr(yarn_config.model.model_config, "block_size", 1024)
        else:
            self.original_max_seq_len = getattr(yarn_config.model.model_config, "block_size", 1024)

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
            model_ckpt = f"data/training/{yarn_config.model.model_name}/checkpoint_best_val.pt"
            path = checkpoint_path or (model_ckpt if os.path.exists(model_ckpt) else getattr(yarn_config, "base_checkpoint_path", None))
            if path and os.path.exists(path):
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                if isinstance(ckpt, dict):
                    ckpt.pop("optimizer_state_dict", None)
                    ckpt.pop("optimizer", None)
                    if yarn_config.original_max_seq_len is not None:
                        self.original_max_seq_len = yarn_config.original_max_seq_len
                    elif "starting_base_ckpt_info" in ckpt:
                        self.original_max_seq_len = ckpt["starting_base_ckpt_info"].get("base_seq_len", 1024)
                    else:
                        base_model_name = getattr(yarn_config.model, "model_name", "").replace("_cpt", "").replace("_CPT", "")
                        try:
                            self.original_max_seq_len = get_model_config(base_model_name).seq_len
                        except Exception:
                            self.original_max_seq_len = getattr(yarn_config.model.model_config, "block_size", 1024)
                self.model.load_state_dict(state_dict)
                del ckpt, state_dict
                gc.collect()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def forward_step(self, *args, **kwargs):
        if self.inference_model:
            return self.model.forward_step(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def set_context_length(self, seq_len: int, scale: Optional[float] = None) -> int:
        return patch_model_yarn(
            self.model,
            target_seq_len=seq_len,
            original_seq_len=self.original_max_seq_len,
            beta_fast=float(self.yarn_config.beta_fast),
            beta_slow=float(self.yarn_config.beta_slow),
            dtype=self.dtype,
            scale=scale,
        )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
