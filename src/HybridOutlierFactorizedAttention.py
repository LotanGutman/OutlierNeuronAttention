import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.config import ModelConfig
from src.chunk_gla_inlier import chunk_gla_inlier_fwd
from src.exact_attention import exact_attention_triton
from src.hofa_decode_triton import fused_hofa_decode
from src.modules.modules import RotaryEmbedding, apply_rotary_pos_emb

class HybridOutlierFactorizedAttention(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.d_model   = model_cfg.d_model
        self.num_heads = model_cfg.num_heads
        self.d_head    = model_cfg.d_head
        self.r         = model_cfg.r
        self.j         = self.d_head - self.r
        self.chunk_size = model_cfg.chunk_size

        if self.r > 0 and (self.r & (self.r - 1)) != 0:
            import warnings
            warnings.warn(f"HOFA efficiency warning: r={self.r} is not a power of 2. Triton kernels will pad it to the next power of 2, wasting computation.")

        # Gate projection uses full Q,K (before routing) for stability
        self.gate_proj = nn.Linear(2 * self.d_head, self.num_heads, bias=True)

        nn.init.constant_(self.gate_proj.bias, -1.0)
        nn.init.zeros_(self.gate_proj.weight)
        
        # Architectural Ablations
        self.forced_mha_heads = getattr(model_cfg, 'mha_heads_for_width_split', 0)
        self.fixed_blend_weight = getattr(model_cfg, 'fixed_blend_weight', False)

        # Dynamic mixing gate between exact and linear pathways
        if not self.fixed_blend_weight and self.forced_mha_heads == 0:
            self.mix_proj = nn.Linear(2 * self.d_head, self.num_heads, bias=True)
            nn.init.zeros_(self.mix_proj.weight)
            nn.init.constant_(self.mix_proj.bias, getattr(model_cfg, 'mix_gate_bias_init', 0.0))
        elif self.fixed_blend_weight and self.forced_mha_heads == 0:
            self.fixed_mix = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))

        # Shared projections (no bias, standard for attention)
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # RoPE dedicated strictly to the exact-match routing dimension
        if (self.r > 0 or self.forced_mha_heads > 0) and getattr(model_cfg, 'use_rope', True):
            self.rotary_emb = RotaryEmbedding(dim=self.r if self.r > 0 else self.d_head)

        # Inlier normalization and learned LayerScale for the GLA pathway
        self.inlier_norm = nn.RMSNorm(self.d_head, elementwise_affine=False)
        self.gla_scale = nn.Parameter(torch.ones(1, self.num_heads, 1, self.d_head))

    def _compute_gates_optimized(self, Q, K):
        # Q, K: (B, H, N, d_head)
        W_g = self.gate_proj.weight  # (H, 2*d_head)
        W_g_q, W_g_k = W_g.chunk(2, dim=1)
        
        gate_logits = torch.einsum('bhnf,hf->bhn', Q, W_g_q) + \
                      torch.einsum('bhnf,hf->bhn', K, W_g_k) + \
                      self.gate_proj.bias.view(1, self.num_heads, 1).to(Q.dtype)

        if hasattr(self, 'mix_proj'):
            W_m = self.mix_proj.weight   # (H, 2*d_head)
            W_m_q, W_m_k = W_m.chunk(2, dim=1)
            mix_logits = torch.einsum('bhnf,hf->bhn', Q, W_m_q) + \
                         torch.einsum('bhnf,hf->bhn', K, W_m_k) + \
                         self.mix_proj.bias.view(1, self.num_heads, 1).to(Q.dtype)
            mix_g = torch.sigmoid(mix_logits).unsqueeze(-1)
        elif hasattr(self, 'fixed_mix'):
            mix_g = torch.sigmoid(self.fixed_mix).expand(Q.shape[0], -1, Q.shape[2], -1)
        else:
            mix_g = None

        return gate_logits.unsqueeze(-1), mix_g

    def forward(self, x, return_state=False):
        B, N, D = x.shape
        dtype_in = x.dtype
        scale_factor = self.d_head ** 0.25

        Q = (self.W_q(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        K = (self.W_k(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)

        if self.forced_mha_heads > 0:
            Q_mha, Q_gla = Q.split([self.forced_mha_heads, self.num_heads - self.forced_mha_heads], dim=1)
            K_mha, K_gla = K.split([self.forced_mha_heads, self.num_heads - self.forced_mha_heads], dim=1)
            V_mha, V_gla = V.split([self.forced_mha_heads, self.num_heads - self.forced_mha_heads], dim=1)

            if hasattr(self, 'rotary_emb'):
                cos, sin = self.rotary_emb(N)
                Q_mha, K_mha = apply_rotary_pos_emb(Q_mha, K_mha, cos, sin)

            Y_mha = F.scaled_dot_product_attention(Q_mha, K_mha, V_mha, is_causal=True, scale=1.0)

            W_g = self.gate_proj.weight
            W_g_q, W_g_k = W_g.chunk(2, dim=1)
            W_g_q_gla = W_g_q[self.forced_mha_heads:]
            W_g_k_gla = W_g_k[self.forced_mha_heads:]
            b_g_gla = self.gate_proj.bias[self.forced_mha_heads:]

            gate_logits = torch.einsum('bhnf,hf->bhn', Q_gla, W_g_q_gla) + \
                          torch.einsum('bhnf,hf->bhn', K_gla, W_g_k_gla) + \
                          b_g_gla.view(1, -1, 1).to(dtype_in)
            log_gamma = F.logsigmoid(-gate_logits).squeeze(-1)

            Y_gla, states_out = chunk_gla_inlier_fwd(Q_gla, K_gla, V_gla, log_gamma, 0, self.chunk_size)
            Y_gla_float = Y_gla.float()
            Y_gla_float = self.inlier_norm(Y_gla_float) * self.gla_scale[:, self.forced_mha_heads:]
            Y_gla = Y_gla_float.to(Y_gla.dtype)

            Y_out = torch.cat([Y_mha, Y_gla], dim=1).transpose(1, 2).reshape(B, N, D)
            Y_out = self.out_proj(Y_out).to(dtype_in)

            if return_state:
                cache_O = (K_mha.contiguous(), V_mha.contiguous())
                state_I = states_out[:, :, :self.d_head, :self.d_head]
                cache_I = (state_I, torch.empty_like(state_I))
                return Y_out, cache_O, cache_I
            return Y_out

        # ----- special cases: r = 0 or r = d_head -----fallback to pure GLA
        if self.r == 0:
            gate_logits, _ = self._compute_gates_optimized(Q, K)
            log_gamma = F.logsigmoid(-gate_logits).squeeze(-1)
            Y_I, states_out = chunk_gla_inlier_fwd(Q, K, V, log_gamma, self.r, self.chunk_size)
            del gate_logits, log_gamma
            Y_I_float = Y_I.float()

        if self.r == self.d_head:
            if hasattr(self, 'rotary_emb'):
                cos, sin = self.rotary_emb(N)
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
            Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=1.0)
            Y = Y.transpose(1, 2).reshape(B, N, D)
            Y_out = self.out_proj(Y).to(dtype_in)
            if return_state:
                cache_O = (K.contiguous(), V.contiguous())
                return Y_out, cache_O, None
            return Y_out

        # ----- obtain routing indices -----
        # (Static routing removed indices fetching)

        gate_logits, mix_g = self._compute_gates_optimized(Q, K)
        log_gamma = F.logsigmoid(-gate_logits).squeeze(-1)

        # --- Pre-calculate RoPE (Q and K outlier dimensions) ---
        if hasattr(self, 'rotary_emb'):
            cos, sin = self.rotary_emb(N)
            Q_O = Q[..., :self.r]
            K_O = K[..., :self.r]
            Q_O_rotated, K_O_rotated = apply_rotary_pos_emb(Q_O, K_O, cos, sin)

            # In-place update to prevent massive tensor duplication
            Q[..., :self.r] = Q_O_rotated
            K[..., :self.r] = K_O_rotated

        # ----- outlier exact attention -----
        # Pseudo-Fused Kernel Execution
        # We pass the full Q and K to the kernels along with the routing indices.
        # This completely avoids O(N * d) memory overhead from .gather()

        # Save for cache export before K is consumed by exact kernel
        if return_state:
            K_cache = K[..., :self.r].contiguous()
            V_cache = V.contiguous()

        Y_I, states_out = chunk_gla_inlier_fwd(Q, K, V, log_gamma, self.r, self.chunk_size)
        del gate_logits, log_gamma

        sm_scale = (self.d_head / self.r) ** 0.5

        # Apply inlier norm and scale
        Y_I_float = Y_I.float()
        Y_I_float = self.inlier_norm(Y_I_float) * self.gla_scale
        Y_I = Y_I_float.to(Y_I.dtype)

        # Pre-allocate Final Output and blend the pre-scaled Y_I
        Y_Final = ((1.0 - mix_g) * Y_I)

        # Run exact attention, adding mix_g * exact directly into Y_Final via True Kernel Fusion
        exact_attention_triton(Q, K, V, self.r, sm_scale, mix_g.squeeze(-1), out=Y_Final)
        del Q, K, V, Y_I

        Y = Y_Final

        # Merge heads
        Y = Y.transpose(1, 2).reshape(B, N, self.d_model)
        Y_out = self.out_proj(Y).to(dtype_in)
        if return_state:
            # Native contiguous FP32 state right from step 0
            state_I = states_out[:, :, :self.j, :self.d_head].contiguous()
            state_I_out = torch.empty_like(state_I)
            return Y_out, (K_cache, V_cache), (state_I, state_I_out)
        return Y_out

    def forward_step(self, x, cache_O=None, state_I=None, cache_seq_len=None):
        """Single - step autoregressive decoding."""
        B, N, D = x.shape
        assert N == 1, "forward_step expects a single token (N=1)"
        dtype_in = x.dtype
        scale_factor = self.d_head ** 0.25

        Q = (self.W_q(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        K = (self.W_k(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)

        if self.forced_mha_heads > 0:
            Q_mha, Q_gla = Q.split([self.forced_mha_heads, self.num_heads - self.forced_mha_heads], dim=1)
            K_mha, K_gla = K.split([self.forced_mha_heads, self.num_heads - self.forced_mha_heads], dim=1)
            V_mha, V_gla = V.split([self.forced_mha_heads, self.num_heads - self.forced_mha_heads], dim=1)

            if hasattr(self, 'rotary_emb'):
                seq_idx = cache_seq_len if cache_seq_len is not None else (cache_O[0].shape[2] if cache_O is not None else 0)
                cos, sin = self.rotary_emb(seq_idx + 1)
                cos = cos[-1:, :]
                sin = sin[-1:, :]
                Q_mha, K_mha = apply_rotary_pos_emb(Q_mha, K_mha, cos, sin)

            if cache_O is None:
                K_past, V_past = K_mha, V_mha
                cache_new = (K_past, V_past)
            elif cache_seq_len is not None and cache_O[0].shape[2] > cache_seq_len:
                cache_O[0][:, :, cache_seq_len:cache_seq_len+1, :] = K_mha
                cache_O[1][:, :, cache_seq_len:cache_seq_len+1, :] = V_mha
                K_past = cache_O[0][:, :, :cache_seq_len+1, :]
                V_past = cache_O[1][:, :, :cache_seq_len+1, :]
                cache_new = cache_O
            else:
                K_past = torch.cat([cache_O[0], K_mha], dim=2)
                V_past = torch.cat([cache_O[1], V_mha], dim=2)
                cache_new = (K_past, V_past)
            Y_mha = F.scaled_dot_product_attention(Q_mha, K_past, V_past, is_causal=False, scale=1.0)

            W_g = self.gate_proj.weight
            W_g_q, W_g_k = W_g.chunk(2, dim=1)
            W_g_q_gla = W_g_q[self.forced_mha_heads:]
            W_g_k_gla = W_g_k[self.forced_mha_heads:]
            b_g_gla = self.gate_proj.bias[self.forced_mha_heads:]

            gate_logits = torch.einsum('bhnf,hf->bhn', Q_gla, W_g_q_gla) + \
                          torch.einsum('bhnf,hf->bhn', K_gla, W_g_k_gla) + \
                          b_g_gla.view(1, -1, 1).to(dtype_in)
            gamma = torch.sigmoid(-gate_logits).view(B, self.num_heads - self.forced_mha_heads, 1, 1)

            if cache_I is None:
                state_I = torch.zeros(B, self.num_heads - self.forced_mha_heads, self.d_head, self.d_head,
                                      device=x.device, dtype=torch.float32)
            else:
                state_I = cache_I[0]

            K_s = K_gla.squeeze(2).to(torch.float32)
            V_s = V_gla.squeeze(2).to(torch.float32)
            q_bmm = Q_gla.squeeze(2).to(torch.float32).reshape(B * (self.num_heads - self.forced_mha_heads), 1, self.d_head)
            k_bmm = K_s.reshape(B * (self.num_heads - self.forced_mha_heads), self.d_head, 1)
            v_bmm = V_s.reshape(B * (self.num_heads - self.forced_mha_heads), 1, self.d_head)
            state_I_bmm = state_I.reshape(B * (self.num_heads - self.forced_mha_heads), self.d_head, self.d_head)
            gamma_bmm = gamma.reshape(B * (self.num_heads - self.forced_mha_heads), 1, 1)

            state_I_new_bmm = torch.baddbmm(state_I_bmm * gamma_bmm, k_bmm, v_bmm)
            Y_gla = torch.bmm(q_bmm, state_I_new_bmm).reshape(B, self.num_heads - self.forced_mha_heads, 1, self.d_head)
            Y_gla_float = Y_gla.float()
            Y_gla_float = self.inlier_norm(Y_gla_float) * self.gla_scale[:, self.forced_mha_heads:]
            Y_gla = Y_gla_float.to(dtype_in)
            state_I_new = state_I_new_bmm.reshape(B, self.num_heads - self.forced_mha_heads, self.d_head, self.d_head)

            Y_out = torch.cat([Y_mha, Y_gla], dim=1).transpose(1, 2).reshape(B, N, D)
            return self.out_proj(Y_out).to(dtype_in), cache_new, (state_I_new, torch.empty_like(state_I_new))

        if self.r == 0:
            gate_logits, _ = self._compute_gates_optimized(Q, K)
            gamma = torch.sigmoid(-gate_logits).view(B, self.num_heads, 1, 1)
            if isinstance(state_I, tuple):
                state_I = state_I[0]
            if state_I is None:
                state_I = torch.zeros(B, self.num_heads, self.j, self.d_head,
                                      device=x.device, dtype=torch.float32)
            K_s = K.squeeze(2)
            V_s = V.squeeze(2)
            
            # --- Causal Fix: compute output with old state, then update state ---
            # Y_I = torch.einsum('bhj,bhjd->bhd', Q.squeeze(2), state_I).unsqueeze(2)
            # state_I_new = gamma * state_I + torch.einsum('bhj,bhd->bhjd', K_s, V_s)
            
            q_bmm = Q.squeeze(2).reshape(B * self.num_heads, 1, self.j)
            k_bmm = K_s.reshape(B * self.num_heads, self.j, 1)
            v_bmm = V_s.reshape(B * self.num_heads, 1, self.d_head)
            state_I_bmm = state_I.reshape(B * self.num_heads, self.j, self.d_head)
            gamma_bmm = gamma.reshape(B * self.num_heads, 1, 1)

            # Inclusive Causality: Update state FIRST (matches training kernel's >= mask)
            state_I_new_bmm = torch.baddbmm(state_I_bmm * gamma_bmm, k_bmm, v_bmm)

            # Compute output using the newly updated state
            Y_I = torch.bmm(q_bmm, state_I_new_bmm).reshape(B, self.num_heads, 1, self.d_head)
            Y_I_float = Y_I.float()
            Y_I_float = self.inlier_norm(Y_I_float) * self.gla_scale
            Y_I = Y_I_float.to(Y_I.dtype)

            state_I_new = state_I_new_bmm.reshape(B, self.num_heads, self.j, self.d_head)
            
            Y_out = Y_I.transpose(1, 2).reshape(B, N, D)
            return self.out_proj(Y_out).to(dtype_in), None, (state_I_new, torch.empty_like(state_I_new))

        if self.r == self.d_head:
            if hasattr(self, 'rotary_emb'):
                seq_idx = cache_seq_len if cache_seq_len is not None else (cache_O[0].shape[2] if cache_O is not None else 0)
                cos, sin = self.rotary_emb(seq_idx + 1)
                cos = cos[-1:, :]
                sin = sin[-1:, :]
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
                
            if cache_O is None:
                K_past, V_past = K, V
                cache_new = (K_past, V_past)
            elif cache_seq_len is not None and cache_O[0].shape[2] > cache_seq_len:
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
        # Compute gates BEFORE rotating Q and K
        gate_logits, mix_g = self._compute_gates_optimized(Q, K)
        log_gamma = F.logsigmoid(-gate_logits).squeeze(-1) # (B, H, N)
        if mix_g is not None:
            mix_g = mix_g.squeeze(-1) # (B, H, N)
        
        if hasattr(self, 'rotary_emb'):
            # Determine correct sequence length for RoPE
            seq_idx = cache_seq_len if cache_seq_len is not None else (cache_O[0].shape[2] if cache_O is not None else 0)
            cos, sin = self.rotary_emb(seq_idx + 1)
            # Only slice the last token for RoPE
            cos = cos[-1:, :]
            sin = sin[-1:, :]
            
            Q_O, K_O = apply_rotary_pos_emb(Q[..., :self.r], K[..., :self.r], cos, sin)
            Q[..., :self.r] = Q_O
            K[..., :self.r] = K_O

        # Append to Cache
        if cache_O is None:
            K_past, V_past = K[..., :self.r], V
            cache_O_new = (K_past, V_past)
            cache_seq_len = 0
        elif cache_seq_len is not None and cache_O[0].shape[2] > cache_seq_len:
            cache_O[0][:, :, cache_seq_len:cache_seq_len+1, :] = K[..., :self.r]
            cache_O[1][:, :, cache_seq_len:cache_seq_len+1, :] = V
            K_past = cache_O[0]
            V_past = cache_O[1]
            cache_O_new = cache_O
        else:
            K_past = torch.cat([cache_O[0], K[..., :self.r]], dim=2)
            V_past = torch.cat([cache_O[1], V], dim=2)
            cache_O_new = (K_past, V_past)
            cache_seq_len = K_past.shape[2] - 1

        if state_I is None:
            state_I = (
                torch.zeros(B, self.num_heads, self.j, self.d_head, device=x.device, dtype=torch.float32),
                torch.empty(B, self.num_heads, self.j, self.d_head, device=x.device, dtype=torch.float32)
            )
        
        state_in, state_out = state_I
        
        # Strict assertions to prove state integrity
        assert state_in.dtype == torch.float32, f"Expected state_in to be float32, got {state_in.dtype}"
        assert state_in.is_contiguous(), "Expected state_in to be contiguous"

        sm_scale = (self.d_head / self.r) ** 0.5
        
        norm_w = self.gla_scale.view(self.num_heads, self.d_head)
        
        Y_out = fused_hofa_decode(
            Q, K, V, 
            cache_O_new[0], cache_O_new[1], 
            state_in, state_out,
            log_gamma, mix_g,
            norm_w,
            self.r,
            cache_seq_len + 1,
            sm_scale
        )

        Y_out = Y_out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(Y_out).to(dtype_in), cache_O_new, (state_out, state_in)


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

    def forward(self, x, return_state=False):
        if return_state:
            attn_out, cache_O, state_I = self.attn(self.ln_1(x), return_state=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, cache_O, state_I
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_step(self, x, cache_O=None, state_I=None, cache_seq_len=None):
        x_attn, new_cache_O, new_state_I = self.attn.forward_step(self.ln_1(x), cache_O, state_I, cache_seq_len)
        x = x + x_attn
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache_O, new_state_I


class SubwordLM(nn.Module):
    def __init__(self, vocab_size: int, model_cfg: ModelConfig):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, model_cfg.d_model)
        
        r_list = model_cfg.r if isinstance(model_cfg.r, (list, tuple)) else [model_cfg.r] * model_cfg.num_layers
        assert len(r_list) == model_cfg.num_layers, f"Length of r ({len(r_list)}) must match num_layers ({model_cfg.num_layers})"
        
        import copy
        self.layers = nn.ModuleList()
        for layer_idx in range(model_cfg.num_layers):
            layer_cfg = copy.copy(model_cfg)
            layer_cfg.r = r_list[layer_idx]
            self.layers.append(TransformerBlock(layer_cfg))
            
        self.ln_f = nn.LayerNorm(model_cfg.d_model)
        self.lm_head = nn.Linear(model_cfg.d_model, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight

    def forward(self, x, targets=None, return_state=False):
        B, N = x.shape
        x = self.token_emb(x)
        if return_state:
            cache_O_list = []
            state_I_list = []
            for layer in self.layers:
                x, c, s = layer(x, return_state=True)
                cache_O_list.append(c)
                state_I_list.append(s)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            return logits, cache_O_list, state_I_list
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def forward_step(self, x, cache_O_list=None, state_I_list=None, cache_seq_len=None):
        B, N = x.shape
        x = self.token_emb(x)
        
        new_cache_O_list = []
        new_state_I_list = []
        
        for i, layer in enumerate(self.layers):
            c_O = cache_O_list[i] if cache_O_list is not None else None
            s_I = state_I_list[i] if state_I_list is not None else None
            
            x, nc_O, ns_I = layer.forward_step(x, c_O, s_I, cache_seq_len)
            
            new_cache_O_list.append(nc_O)
            new_state_I_list.append(ns_I)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_cache_O_list, new_state_I_list

