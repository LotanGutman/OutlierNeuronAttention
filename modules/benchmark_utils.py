import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import numpy as np

from src.HybridOutlierFactorizedAttentionTrain import HybridOutlierFactorizedAttention
from fla.layers import DeltaNet, GatedLinearAttention as GLA
from mamba_ssm import Mamba2
from modules.modules import RotaryEmbedding, apply_rotary_pos_emb

def generate_sniah(batch_size, seq_len, vocab_size, depth_pct, device):
    num_keys = 128
    num_vals = 128
    seq = torch.randint(num_keys + num_vals + 1, vocab_size, (batch_size, seq_len), device=device)
    
    needle_key = torch.randint(1, num_keys + 1, (batch_size, 1), device=device)
    needle_value = torch.randint(num_keys + 1, num_keys + num_vals + 1, (batch_size, 1), device=device)
    
    needle_idx = int(depth_pct * (seq_len // 2 - 2)) * 2
    seq[:, needle_idx:needle_idx+1] = needle_key
    seq[:, needle_idx+1:needle_idx+2] = needle_value
    
    x = seq.clone()
    x[:, -1:] = needle_key
    y = torch.full_like(x, -100)
    y[:, -1:] = needle_value
    return x, y


class StandardMHA(nn.Module):
    def __init__(self, d_model, num_heads, use_rope=False):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.use_rope = use_rope
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(dim=self.d_head)

    def forward(self, x):
        B, N, D = x.shape
        q = self.W_q(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        
        if self.use_rope:
            cos, sin = self.rotary_emb(N)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(y)

class FLAWrapper(nn.Module):
    def __init__(self, fla_layer):
        super().__init__()
        self.layer = fla_layer

    def forward(self, x):
        out = self.layer(x)
        return out[0] if isinstance(out, tuple) else out

class AttentionType(Enum):
    MHA = "mha"
    HOFA = "hofa"
    DELTA = "delta"
    GLA = "gla"
    MAMBA = "mamba"

def build_attention(attn_type, model_cfg):
    d_model = model_cfg.d_model
    num_heads = model_cfg.num_heads
    if attn_type == AttentionType.MHA:
        return StandardMHA(d_model, num_heads, getattr(model_cfg, 'use_rope', False))
    if attn_type == AttentionType.HOFA:
        return HybridOutlierFactorizedAttention(model_cfg)
    if attn_type == AttentionType.DELTA:
        return FLAWrapper(DeltaNet(hidden_size=d_model, num_heads=num_heads))
    if attn_type == AttentionType.GLA:
        return FLAWrapper(GLA(hidden_size=d_model, num_heads=num_heads))
    if attn_type == AttentionType.MAMBA:
        return Mamba2(
            d_model=d_model,
            d_state=2 ** int(np.log2(d_model // num_heads)),
            d_conv=4,
            expand=2
        )
    raise ValueError(f"Unknown attention type: {attn_type}")

class GenericBenchmarkLM(nn.Module):
    def __init__(self, vocab_size, d_model, attn_type, num_heads=8, num_layers=2, model_cfg=None):
        super().__init__()
        self.use_rope = getattr(model_cfg, 'use_rope', False) if model_cfg else False
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        if not self.use_rope:
            self.pos_embedding = nn.Embedding(131072, d_model)
        else:
            self.pos_embedding = None

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln_1': nn.LayerNorm(d_model),
                'attn': build_attention(attn_type, model_cfg),
                'ln_2': nn.LayerNorm(d_model),
                'mlp': nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
            }) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight

    def forward(self, x, targets=None, return_loss=True):
        x = self.token_embedding(x)
        if self.pos_embedding is not None:
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
            x = x + self.pos_embedding(positions)

        for block in self.blocks:
            x = x + block['attn'](block['ln_1'](x))
            x = x + block['mlp'](block['ln_2'](x))
            
        x = self.ln_f(x)
        
        if targets is not None:
            valid_mask = (targets != -100)
            x_valid = x[valid_mask]
            logits_valid = self.lm_head(x_valid)
            
            if return_loss:
                targets_valid = targets[valid_mask]
                loss = F.cross_entropy(logits_valid, targets_valid)
                return None, loss
            else:
                return logits_valid, None
        else:
            logits = self.lm_head(x)
            return logits, None


def adjust_learning_rate(optimizer, step, total_steps, base_lr, warmup_steps):
    lr = base_lr * min(1.0, step / warmup_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
