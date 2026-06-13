import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import ModelConfig
from fla.ops.gla import chunk_gla


class HybridOutlierFactorizedAttention(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.d_model = model_cfg.d_model
        self.num_heads = model_cfg.num_heads
        self.d_head = model_cfg.d_head
        self.r = model_cfg.r
        self.j = self.d_head - self.r

        # Gate projection uses full Q,K (before routing) for stability
        self.gate_proj = nn.Linear(2 * self.d_head, self.num_heads, bias=True)

        nn.init.constant_(self.gate_proj.bias, -1.0)
        nn.init.zeros_(self.gate_proj.weight)

        # Shared projections (no bias, standard for attention)
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Learnable weighting between pathways (per head)
        self.alpha = nn.Parameter(torch.full((self.num_heads,), 0.5))

        # RMSNorm to balance magnitudes of the linear attention pathway
        self.inlier_norm = nn.RMSNorm(self.d_head)

    def _compute_gate_logits(self, Q, K):
        """
        Q, K: (B, H, N, d_head)   full features (before routing)
        Returns gate logits: (B, H, N, 1)  (scalar per head per token)
        """
        qk = torch.cat([Q, K], dim=-1)  # (B, H, N, 2*d_head)
        logits = torch.einsum('bhnf,hf->bhn', qk, self.gate_proj.weight) + \
                 self.gate_proj.bias.view(1, self.num_heads, 1)
        return logits.unsqueeze(-1)  # (B, H, N, 1)

    def forward(self, x):
        B, N, D = x.shape
        dtype_in = x.dtype
        scale_factor = self.d_head ** 0.25

        Q = (self.W_q(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        K = (self.W_k(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)

        # ----- special cases: r = 0 or r = d_head -----
        if self.r == 0:
            gate_logits = self._compute_gate_logits(Q, K)
            log_gamma = F.logsigmoid(-gate_logits)
            log_gamma = log_gamma.expand(-1, -1, -1, self.d_head)
            Y_I, _ = chunk_gla(Q, K, V, g=log_gamma, scale=1.0, output_final_state=False)
            Y_out = Y_I.transpose(1, 2).reshape(B, N, D)
            return self.out_proj(Y_out).to(dtype_in)

        if self.r == self.d_head:
            Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=1.0)
            Y = Y.transpose(1, 2).reshape(B, N, D)
            return self.out_proj(Y).to(dtype_in)

        gate_logits = self._compute_gate_logits(Q, K)
        log_gamma = F.logsigmoid(-gate_logits)
        log_gamma = log_gamma.expand(-1, -1, -1, self.d_head)

        # ----- outlier exact attention -----
        Q_O = Q[..., :self.r]
        K_O = K[..., :self.r]
        Y_O = F.scaled_dot_product_attention(Q_O, K_O, V, is_causal=True, scale=1.0)

        # ----- inlier gated linear attention -----
        Q_J = Q[..., self.r:]
        K_J = K[..., self.r:]

        Q_J = F.pad(Q_J, (0, self.r))
        K_J = F.pad(K_J, (0, self.r))

        Y_I, _ = chunk_gla(Q_J, K_J, V, g=log_gamma, scale=1.0, output_final_state=False)

        alpha = self.alpha.view(1, self.num_heads, 1, 1)
        Y_out = (alpha * Y_O) + ((1.0 - alpha) * self.inlier_norm(Y_I))
        Y_out = Y_out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(Y_out).to(dtype_in)

    def forward_step(self, x, cache_O=None, state_I=None):
        """Single-step autoregressive decoding."""
        B, N, D = x.shape
        assert N == 1, "forward_step expects a single token (N=1)"
        dtype_in = x.dtype
        scale_factor = self.d_head ** 0.25

        Q = (self.W_q(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        K = (self.W_k(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)

        if self.r == 0:
            gate_logits = self._compute_gate_logits(Q, K)
            gamma = torch.sigmoid(-gate_logits).view(B, self.num_heads, 1, 1)
            if state_I is None:
                state_I = torch.zeros(B, self.num_heads, self.d_head, self.d_head,
                                      device=x.device, dtype=x.dtype)
            K_s = K.squeeze(2)
            V_s = V.squeeze(2)

            Y_I = torch.einsum('bhi,bhij->bhj', Q.squeeze(2), state_I).unsqueeze(2)
            state_I_new = gamma * state_I + torch.einsum('bhi,bhj->bhij', K_s, V_s)

            Y_out = Y_I.transpose(1, 2).reshape(B, N, D)
            return self.out_proj(Y_out).to(dtype_in), None, state_I_new

        if self.r == self.d_head:
            if cache_O is None:
                K_past, V_past = K, V
            else:
                K_past = torch.cat([cache_O[0], K], dim=2)
                V_past = torch.cat([cache_O[1], V], dim=2)
            Y = F.scaled_dot_product_attention(Q, K_past, V_past, is_causal=False, scale=1.0)
            cache_new = (K_past, V_past)
            Y_out = Y.transpose(1, 2).reshape(B, N, D)
            return self.out_proj(Y_out).to(dtype_in), cache_new, None

        gate_logits = self._compute_gate_logits(Q, K)

        # ----- outlier exact attention -----
        Q_O = Q[..., :self.r]
        K_O = K[..., :self.r]

        if cache_O is None:
            cache_O = (K_O, V)
        else:
            K_cache, V_cache = cache_O
            K_cache = torch.cat([K_cache, K_O], dim=2)
            V_cache = torch.cat([V_cache, V], dim=2)
            cache_O = (K_cache, V_cache)

        attn_weights = torch.einsum('bhid,bhjd->bhij', Q_O, cache_O[0])
        Y_O = torch.einsum('bhij,bhjd->bhid', torch.softmax(attn_weights, dim=-1), cache_O[1])

        # ----- inlier gated linear attention -----
        Q_J = Q[..., self.r:]
        K_J = K[..., self.r:]

        Q_J = F.pad(Q_J, (0, self.r))
        K_J = F.pad(K_J, (0, self.r))

        gamma = torch.sigmoid(-gate_logits).view(B, self.num_heads, 1, 1)

        if state_I is None:
            state_I = torch.zeros(B, self.num_heads, self.d_head, self.d_head,
                                  device=x.device, dtype=x.dtype)

        K_J_s = K_J.squeeze(2)
        V_s = V.squeeze(2)

        state_I = state_I * gamma + torch.einsum('bhd,bhm->bhdm', K_J_s, V_s)
        Y_I = torch.einsum('bhd,bhdm->bhm', Q_J.squeeze(2), state_I).unsqueeze(2)

        alpha = self.alpha.view(1, self.num_heads, 1, 1)
        Y_out = (alpha * Y_O) + ((1.0 - alpha) * self.inlier_norm(Y_I))
        Y_out = Y_out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(Y_out).to(dtype_in), cache_O, state_I


class TransformerBlock(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(model_cfg.d_model)
        self.attn = HybridOutlierFactorizedAttention(model_cfg)
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
            TransformerBlock(model_cfg) for _ in range(model_cfg.num_layers)
        ])
        self.ln_f = nn.LayerNorm(model_cfg.d_model)
        self.lm_head = nn.Linear(model_cfg.d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight

    def forward(self, x, targets=None):
        B, N = x.shape
        pos = torch.arange(0, N, dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
