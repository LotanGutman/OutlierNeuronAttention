import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import ModelConfig
from src.chunk_gla_inlier import ChunkGLAInlier
from src.exact_attention import exact_attention_triton
from src.chunk_gla_inlier import ChunkGLAInlier

class HybridOutlierFactorizedAttention(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.d_model   = model_cfg.d_model
        self.num_heads = model_cfg.num_heads
        self.d_head    = model_cfg.d_head
        self.r         = model_cfg.r
        self.j         = self.d_head - self.r
        self.chunk_size = model_cfg.chunk_size

        # Gate projection uses full Q,K (before routing) for stability
        self.gate_proj = nn.Linear(2 * self.d_head, self.num_heads, bias=True)

        nn.init.constant_(self.gate_proj.bias, -1.0)
        nn.init.zeros_(self.gate_proj.weight)
        
        # Learnable weighting between pathways
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Shared projections (no bias, standard for attention)
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Outlier routing indices – cached and refreshed every _refresh_steps steps
        self.register_buffer('_cached_outlier_idx', None)
        self.register_buffer('_cached_inlier_idx', None)
        self.register_buffer('_step_counter', torch.tensor(0, dtype=torch.long))

        # Refresh period – pull from config if available, otherwise default to 100
        refresh_steps = getattr(model_cfg, 'refresh_steps', 100)
        self.register_buffer('_refresh_steps', torch.tensor(refresh_steps, dtype=torch.long))

    @torch.no_grad()
    def get_routing_indices(self):
        wq = self.W_q.weight.view(self.num_heads, self.d_head, self.d_model)
        wk = self.W_k.weight.view(self.num_heads, self.d_head, self.d_model)
        wv = self.W_v.weight.view(self.num_heads, self.d_head, self.d_model)
        score = (wq.norm(dim=-1) * wk.norm(dim=-1) * wv.norm(dim=-1))
        _, out_idx = torch.topk(score, self.r, dim=-1)
        mask = torch.ones(self.num_heads, self.d_head, dtype=torch.bool, device=wq.device)
        mask.scatter_(1, out_idx, False)
        in_idx = mask.nonzero()[:, 1].view(self.num_heads, self.j)
        return out_idx, in_idx

    def _maybe_update_indices(self, increment=True):
        """Optionally increment the step counter and refresh routing indices if needed."""
        if increment:
            self._step_counter.add_(1)

        if self._cached_outlier_idx is None:
            o, i = self.get_routing_indices()
            self._cached_outlier_idx = o
            self._cached_inlier_idx = i
        elif increment and (self._step_counter.item() % self._refresh_steps.item() == 0):
            o, i = self.get_routing_indices()
            self._cached_outlier_idx = o
            self._cached_inlier_idx = i

    def _compute_gate_logits(self, Q, K):
        """
        Q, K: (B, H, N, d_head)   full features (before routing)
        Returns gate logits: (B, H, N, 1)  (scalar per head per token)
        """
        qk = torch.cat([Q, K], dim=-1)                      # (B, H, N, 2*d_head)
        logits = torch.einsum('bhnf,hf->bhn', qk, self.gate_proj.weight) + \
                 self.gate_proj.bias.view(1, self.num_heads, 1)
        return logits.unsqueeze(-1)                         # (B, H, N, 1)

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
            log_gamma = F.logsigmoid(-gate_logits).squeeze(-1)
            Y_I = ChunkGLAInlier.apply(Q, K, V, log_gamma, self.chunk_size) # this would error if r=0 was actually called because missing inlier_idx but user said do not fix
            del gate_logits, log_gamma
            Y_out = Y_I.transpose(1, 2).reshape(B, N, D)
            return self.out_proj(Y_out).to(dtype_in)

        if self.r == self.d_head:
            Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=1.0)
            Y = Y.transpose(1, 2).reshape(B, N, D)
            return self.out_proj(Y).to(dtype_in)

        # ----- obtain routing indices -----
        self._maybe_update_indices(increment=True)
        outlier_idx = self._cached_outlier_idx
        inlier_idx  = self._cached_inlier_idx

        gate_logits = self._compute_gate_logits(Q, K)
        log_gamma = F.logsigmoid(-gate_logits).squeeze(-1)

        # ----- outlier exact attention -----
        # Pseudo-Fused Kernel Execution
        # We pass the full Q and K to the kernels along with the routing indices.
        # This completely avoids O(N * d) memory overhead from .gather()
        
        Y_I = ChunkGLAInlier.apply(Q, K, V, log_gamma, inlier_idx, self.chunk_size)
        del gate_logits, log_gamma
        
        Y_O = exact_attention_triton(Q, K, V, outlier_idx, out=None)
        del Q, K, V, outlier_idx, inlier_idx
        
        Y = self.alpha * Y_O + (1 - self.alpha) * Y_I
        
        # Merge heads
        Y = Y.transpose(1, 2).reshape(B, N, self.d_model)
        return self.out_proj(Y).to(dtype_in)

    def forward_step(self, x, cache_O=None, state_I=None, cache_seq_len=None):
        """Single - step autoregressive decoding."""
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
                state_I = torch.zeros(B, self.num_heads, self.j, self.d_head,
                                      device=x.device, dtype=x.dtype)
            K_s = K.squeeze(2)
            V_s = V.squeeze(2)
            
            # --- Causal Fix: compute output with old state, then update state ---
            # Y_I = torch.einsum('bhj,bhjd->bhd', Q.squeeze(2), state_I).unsqueeze(2)
            # state_I_new = gamma * state_I + torch.einsum('bhj,bhd->bhjd', K_s, V_s)
            
            q_bmm = Q.squeeze(2).view(B * self.num_heads, 1, self.j)
            k_bmm = K_s.view(B * self.num_heads, self.j, 1)
            v_bmm = V_s.view(B * self.num_heads, 1, self.d_head)
            state_I_bmm = state_I.view(B * self.num_heads, self.j, self.d_head)
            gamma_bmm = gamma.view(B * self.num_heads, 1, 1)

            Y_I = torch.bmm(q_bmm, state_I_bmm).view(B, self.num_heads, 1, self.d_head)
            state_I_new_bmm = torch.baddbmm(state_I_bmm * gamma_bmm, k_bmm, v_bmm)
            state_I_new = state_I_new_bmm.view(B, self.num_heads, self.j, self.d_head)
            
            Y_out = Y_I.transpose(1, 2).reshape(B, N, D)
            return self.out_proj(Y_out).to(dtype_in), None, state_I_new

        if self.r == self.d_head:
            if cache_O is None:
                K_past, V_past = K, V
                cache_new = (K_past, V_past)
            elif cache_seq_len is not None:
                cache_O[0][:, :, cache_seq_len:cache_seq_len+1, :] = K
                cache_O[1][:, :, cache_seq_len:cache_seq_len+1, :] = V
                K_past = cache_O[0][:, :, :cache_seq_len+1, :]
                V_past = cache_O[1][:, :, :cache_seq_len+1, :]
                cache_new = cache_O
            else:
                K_past = torch.cat([cache_O[0], K], dim=2)
                V_past = torch.cat([cache_O[1], V], dim=2)
                cache_new = (K_past, V_past)
            Y = F.scaled_dot_product_attention(Q, K_past, V_past, is_causal=False, scale=1.0)
            Y_out = Y.transpose(1, 2).reshape(B, N, D)
            return self.out_proj(Y_out).to(dtype_in), cache_new, None

        # ----- hybrid step -----
        self._maybe_update_indices(increment=False)
        outlier_idx = self._cached_outlier_idx
        inlier_idx  = self._cached_inlier_idx

        # Append to Cache
        if cache_O is None:
            K_past, V_past = K, V
            cache_O_new = (K_past, V_past)
            cache_seq_len = 0
        elif cache_seq_len is not None:
            cache_O[0][:, :, cache_seq_len:cache_seq_len+1, :] = K
            cache_O[1][:, :, cache_seq_len:cache_seq_len+1, :] = V
            K_past = cache_O[0]
            V_past = cache_O[1]
            cache_O_new = cache_O
        else:
            K_past = torch.cat([cache_O[0], K], dim=2)
            V_past = torch.cat([cache_O[1], V], dim=2)
            cache_O_new = (K_past, V_past)
            cache_seq_len = K_past.shape[2] - 1

        seq_len = cache_seq_len + 1

        if state_I is None:
            state_I = torch.zeros(B, self.num_heads, self.j, self.d_head,
                                  device=x.device, dtype=x.dtype)

        from src.hofa_decode_triton import fused_hofa_decode
        Y_out, state_I_new = fused_hofa_decode(
            Q, K, V, K_past, V_past, state_I,
            outlier_idx, inlier_idx,
            self.gate_proj.weight, self.gate_proj.bias,
            self.alpha, seq_len
        )

        Y_out = Y_out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(Y_out).to(dtype_in), cache_O_new, state_I_new


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
