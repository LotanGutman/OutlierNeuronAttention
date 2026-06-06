import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from src.config import ModelConfig

# ---------- Attention ----------
class OutlierFactorizedLinearAttention(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.d_model = model_cfg.d_model
        self.num_heads = model_cfg.num_heads
        self.d_head = model_cfg.d_head
        self.r = model_cfg.r
        self.m = model_cfg.m
        self.m_O = model_cfg.m_O
        self.j = self.d_head - self.r
        self.chunk_size = model_cfg.chunk_size   # kept for compatibility, not used in training

        self.register_buffer('omega_J', torch.randn(self.j, self.m))
        self.register_buffer('omega_O', torch.randn(self.r, self.m_O))

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)

        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.register_buffer('_cached_outlier_idx', None)
        self.register_buffer('_cached_inlier_idx', None)
        self._step_counter = 0
        self._refresh_steps = 100
        # self.forward = torch.compile(self.forward, mode='reduce-overhead')


    @torch.no_grad()
    def get_routing_indices(self):
        w_q = self.W_q.weight.view(self.num_heads, self.d_head, self.d_model)
        w_k = self.W_k.weight.view(self.num_heads, self.d_head, self.d_model)
        w_v = self.W_v.weight.view(self.num_heads, self.d_head, self.d_model)
        q_norms = torch.norm(w_q, p=2, dim=-1)
        k_norms = torch.norm(w_k, p=2, dim=-1)
        v_norms = torch.norm(w_v, p=2, dim=-1)
        outlier_score = q_norms * k_norms * v_norms
        _, outlier_idx = torch.topk(outlier_score, self.r, dim=-1)

        mask = torch.ones(self.num_heads, self.d_head, dtype=torch.bool, device=w_q.device)
        mask.scatter_(1, outlier_idx, False)
        inlier_idx = mask.nonzero()[:, 1].view(self.num_heads, self.j)
        
        return outlier_idx, inlier_idx

    def _maybe_update_indices(self):
        self._step_counter += 1
        if self._cached_outlier_idx is None or self._step_counter % self._refresh_steps == 0:
            o, i = self.get_routing_indices()
            self._cached_outlier_idx = o
            self._cached_inlier_idx = i

    def forward(self, x):
        B, N, D = x.shape
        dtype_in = x.dtype
        scale_factor = self.d_head ** 0.25
        inlier_scale = float(self.j ** 0.5)        # <-- new: stabilises E

        Q = (self.W_q(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        K = (self.W_k(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)

        # ---- Special case: r == 0 → pure causal linear attention ----
        if self.r == 0:
            phi_Q = torch.exp(Q @ self.omega_J.to(dtype_in) -
                              (Q ** 2).sum(dim=-1, keepdim=True) / 2.0) / math.sqrt(self.m)
            phi_K = torch.exp(K @ self.omega_J.to(dtype_in) -
                              (K ** 2).sum(dim=-1, keepdim=True) / 2.0) / math.sqrt(self.m)
            KV = torch.einsum('bhnm, bhnd -> bhnmd', phi_K, V)
            KV_cum = torch.cumsum(KV, dim=2)
            K_cum = torch.cumsum(phi_K, dim=2)
            num = torch.einsum('bhnm, bhnmd -> bhnd', phi_Q, KV_cum)
            den = torch.einsum('bhnm, bhnm -> bhn', phi_Q, K_cum).unsqueeze(-1)
            Y = num / (den + 1e-8)
            
            return Y.transpose(1, 2).reshape(B, N, D)

        # ---- Special case: r == d_head → true exact softmax ----
        if self.r == self.d_head:
            attn = torch.einsum('bhqd,bhkd->bhqk', Q, K)
            causal_mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
            attn = torch.where(causal_mask, attn,
                               torch.tensor(torch.finfo(attn.dtype).min,
                                            device=x.device, dtype=attn.dtype))
            P = F.softmax(attn, dim=-1)
            Y = torch.einsum('bhqk,bhkd->bhqd', P, V)
            
            return Y.transpose(1, 2).reshape(B, N, D)

        self._maybe_update_indices()
        outlier_idx = self._cached_outlier_idx
        inlier_idx = self._cached_inlier_idx

        out_gather = outlier_idx.view(1, self.num_heads, 1, self.r).expand(B, self.num_heads, N, self.r)
        in_gather = inlier_idx.view(1, self.num_heads, 1, self.j).expand(B, self.num_heads, N, self.j)

        Q_O = Q.gather(-1, out_gather)
        K_O = K.gather(-1, out_gather)
        Q_J = Q.gather(-1, in_gather)
        K_J = K.gather(-1, in_gather)

        # --- 1. Outlier exact attention (global) ---
        attn_O = torch.einsum('bhqd,bhkd->bhqk', Q_O, K_O)
        causal_mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
        attn_O = torch.where(causal_mask, attn_O,
                             torch.tensor(torch.finfo(attn_O.dtype).min,
                                          device=x.device, dtype=attn_O.dtype))
        P_O = F.softmax(attn_O, dim=-1)          # (B, H, N, N)
        Y_O = torch.einsum('bhqk,bhkd->bhqd', P_O, V)   # (B, H, N, d_head)

        # --- 2. Inlier correction (global, scaled, clamped) ---
        E = torch.einsum('bhin,bhjn->bhij', Q_J, K_J) / inlier_scale   # (B,H,N,N)
        causal_mask_full = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
        E = E.masked_fill(~causal_mask_full, 0.0)

        e = (P_O * E).sum(dim=-1)                # (B, H, N)
        e = e.clamp(-0.95, 0.95)                 # safety

        term1 = torch.einsum('bhij,bhjd->bhid', P_O * E, V)   # (B, H, N, d_head)

        Y = Y_O * (1.0 - e.unsqueeze(-1)) + term1
        Y = Y.transpose(1, 2).reshape(B, N, D)
        Y = self.out_proj(Y)

        return Y.to(dtype_in)

class TransformerBlock(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(model_cfg.d_model)
        self.attn = OutlierFactorizedLinearAttention(model_cfg)
        self.ln_2 = nn.LayerNorm(model_cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(model_cfg.d_model, 4 * model_cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * model_cfg.d_model, model_cfg.d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SubwordLM(nn.Module):
    def __init__(self, vocab_size: int, model_cfg: ModelConfig):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, model_cfg.d_model)
        self.pos_emb = nn.Embedding(model_cfg.block_size, model_cfg.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(model_cfg)
            for _ in range(model_cfg.num_layers)
        ])
        self.ln_f = nn.LayerNorm(model_cfg.d_model)
        self.lm_head = nn.Linear(model_cfg.d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight

    def forward(self, x, targets=None):
        B, N = x.shape
        pos = torch.arange(0, N, dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
