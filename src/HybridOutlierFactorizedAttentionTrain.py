import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from src.config import ModelConfig
from fla.ops.gla import chunk_gla
from src.modules.modules import RotaryEmbedding, apply_rotary_pos_emb

class HybridOutlierFactorizedAttention(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.d_model = model_cfg.d_model
        self.num_heads = model_cfg.num_heads
        self.d_head = model_cfg.d_head
        self.r = model_cfg.r
        self.j = self.d_head - self.r
        self.chunk_size = model_cfg.chunk_size

        self.is_power_of_2 = (self.r & (self.r - 1)) == 0
        if self.r > 0 and not self.is_power_of_2:
            import warnings
            warnings.warn(f"HOFA efficiency warning: r={self.r} is not a power of 2. Triton kernels will pad it to the next power of 2, wasting computation.")

        # Gate projection uses full Q,K (before routing) for stability
        self.gate_proj = nn.Linear(2 * self.d_head, self.num_heads, bias=True)
        self.gate_proj._is_gate = True

        nn.init.constant_(self.gate_proj.bias, -1.0)
        nn.init.zeros_(self.gate_proj.weight)

        # Shared projections (no bias, standard for attention)
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Dynamic mixing gate between exact and linear pathways
        self.mix_proj = nn.Linear(2 * self.d_head, self.num_heads, bias=True)
        self.mix_proj._is_gate = True
        nn.init.zeros_(self.mix_proj.weight)
        nn.init.constant_(self.mix_proj.bias, 0.0) # Initializes mix_g to 0.5 to allow gradients to both pathways

        # RoPE dedicated strictly to the exact-match routing dimension
        if self.r > 0 and model_cfg.use_rope:
            self.rotary_emb = RotaryEmbedding(dim=self.r)

        # Inlier normalization and learned LayerScale for the GLA pathway
        self.inlier_norm = nn.RMSNorm(self.d_head, elementwise_affine=False)
        self.gla_scale = nn.Parameter(torch.ones(1, self.num_heads, 1, self.d_head))

        std = 0.02
        res_std = std / math.sqrt(2 * model_cfg.num_layers)
        
        nn.init.normal_(self.W_q.weight, std=std)
        nn.init.normal_(self.W_k.weight, std=std)
        nn.init.normal_(self.W_v.weight, std=std)
        nn.init.normal_(self.out_proj.weight, std=res_std)

    def _compute_gates_optimized(self, Q, K):
        # Q, K: (B, H, N, d_head)
        W_g = self.gate_proj.weight  # (H, 2*d_head)
        W_m = self.mix_proj.weight   # (H, 2*d_head)

        W_g_q, W_g_k = W_g.chunk(2, dim=1)
        W_m_q, W_m_k = W_m.chunk(2, dim=1)

        gate_logits = torch.einsum('bhnf,hf->bhn', Q, W_g_q) + \
                      torch.einsum('bhnf,hf->bhn', K, W_g_k) + \
                      self.gate_proj.bias.view(1, self.num_heads, 1).to(Q.dtype)

        mix_logits = torch.einsum('bhnf,hf->bhn', Q, W_m_q) + \
                     torch.einsum('bhnf,hf->bhn', K, W_m_k) + \
                     self.mix_proj.bias.view(1, self.num_heads, 1).to(Q.dtype)

        return gate_logits.unsqueeze(-1), torch.sigmoid(mix_logits).unsqueeze(-1)

    def forward(self, x):
        B, N, D = x.shape
        scale_factor = self.d_head ** 0.25

        Q = (self.W_q(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        # Capture the active dtype AFTER the autocasted linear projection, because 
        # x.dtype is float32 (Embeddings do not autocast in PyTorch)
        dtype_in = Q.dtype
        K = (self.W_k(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)

        # ----- special cases: r = 0 or r = d_head -----
        if self.r == 0:
            gate_logits, _ = self._compute_gates_optimized(Q, K)
            log_gamma = F.logsigmoid(-gate_logits)
            
            q_gla_t = Q.transpose(1, 2).to(torch.float32).contiguous()
            k_gla_t = K.transpose(1, 2).to(torch.float32).contiguous()
            v_gla_t = V.transpose(1, 2).to(torch.float32).contiguous()
            g_gla_t = log_gamma.expand(-1, -1, -1, self.d_head).transpose(1, 2).to(torch.float32).contiguous()
            
            assert q_gla_t.is_contiguous() and q_gla_t.dtype == torch.float32, "q_gla_t must be contiguous float32"
            assert k_gla_t.is_contiguous() and k_gla_t.dtype == torch.float32, "k_gla_t must be contiguous float32"
            assert v_gla_t.is_contiguous() and v_gla_t.dtype == torch.float32, "v_gla_t must be contiguous float32"
            assert g_gla_t.is_contiguous() and g_gla_t.dtype == torch.float32, "g_gla_t must be contiguous float32"

            Y_I_t, _ = chunk_gla(
                q_gla_t, 
                k_gla_t, 
                v_gla_t, 
                g=g_gla_t,
                scale=1.0, 
                output_final_state=False
            )
            Y_I = Y_I_t.transpose(1, 2)
            Y_I = (self.inlier_norm(Y_I.float()) * self.gla_scale).to(dtype_in)

            Y_out = Y_I.transpose(1, 2).reshape(B, N, D)
            return self.out_proj(Y_out)

        if self.r == self.d_head:
            if getattr(self, 'rotary_emb', None) is not None:
                cos, sin = self.rotary_emb(N)
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
            Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=1.0)
            Y = Y.transpose(1, 2).reshape(B, N, D)
            return self.out_proj(Y)

        gate_logits, mix_g = self._compute_gates_optimized(Q, K)
        self.last_mix_g = mix_g.detach()
        log_gamma = F.logsigmoid(-gate_logits)

        # ----- outlier exact attention -----
        Q_O = Q[..., :self.r]
        K_O = K[..., :self.r]

        if getattr(self, 'rotary_emb', None) is not None:
            cos, sin = self.rotary_emb(N)
            Q_O, K_O = apply_rotary_pos_emb(Q_O, K_O, cos, sin)

        sm_scale = (self.d_head / self.r) ** 0.5
        
        if not self.is_power_of_2:
            # MemEfficient attention requires head dim to be a multiple of 8
            pad_len = (8 - (self.r % 8)) % 8
            if pad_len > 0:
                Q_O_padded = F.pad(Q_O, (0, pad_len))
                K_O_padded = F.pad(K_O, (0, pad_len))
            else:
                Q_O_padded, K_O_padded = Q_O, K_O
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                Y_O = F.scaled_dot_product_attention(Q_O_padded, K_O_padded, V, is_causal=True, scale=sm_scale)
        else:
            Y_O = F.scaled_dot_product_attention(Q_O, K_O, V, is_causal=True, scale=sm_scale)

        # ----- inlier gated linear attention -----
        Q_J = Q[..., self.r:]
        K_J = K[..., self.r:]

        q_gla_t = Q_J.transpose(1, 2).to(torch.float32).contiguous()
        k_gla_t = K_J.transpose(1, 2).to(torch.float32).contiguous()
        v_gla_t = V.transpose(1, 2).to(torch.float32).contiguous()
        g_gla_t = log_gamma.expand(-1, -1, -1, K_J.shape[-1]).transpose(1, 2).to(torch.float32).contiguous()
        
        assert q_gla_t.is_contiguous() and q_gla_t.dtype == torch.float32, "q_gla_t must be contiguous float32"
        assert k_gla_t.is_contiguous() and k_gla_t.dtype == torch.float32, "k_gla_t must be contiguous float32"
        assert v_gla_t.is_contiguous() and v_gla_t.dtype == torch.float32, "v_gla_t must be contiguous float32"
        assert g_gla_t.is_contiguous() and g_gla_t.dtype == torch.float32, "g_gla_t must be contiguous float32"

        Y_I_t, _ = chunk_gla(
            q_gla_t, 
            k_gla_t, 
            v_gla_t, 
            g=g_gla_t,
            scale=1.0, 
            output_final_state=False
        )
        Y_I = Y_I_t.transpose(1, 2)
        Y_I = (self.inlier_norm(Y_I.float()) * self.gla_scale).to(dtype_in)

        Y_out = torch.lerp(Y_I, Y_O, mix_g)
        Y_out = Y_out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(Y_out)

    def forward_step(self, x, cache_O=None, state_I=None):
        """Single-step autoregressive decoding."""
        B, N, D = x.shape
        assert N == 1, "forward_step expects a single token (N=1)"
        scale_factor = self.d_head ** 0.25

        Q = (self.W_q(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        dtype_in = Q.dtype
        K = (self.W_k(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)

        if self.r == 0:
            gate_logits, _ = self._compute_gates_optimized(Q, K)
            gamma = torch.sigmoid(-gate_logits).view(B, self.num_heads, 1, 1)
            if state_I is None:
                state_I = torch.zeros(B, self.num_heads, self.d_head, self.d_head,
                                      device=x.device, dtype=torch.float32)
            K_s = K.squeeze(2)
            V_s = V.squeeze(2)

            K_f32 = K_s.to(torch.float32).unsqueeze(-1)
            V_f32 = V_s.to(torch.float32).unsqueeze(-2)
            state_I_new = gamma.to(torch.float32) * state_I + (K_f32 @ V_f32)
            
            Q_f32 = Q.squeeze(2).to(torch.float32).unsqueeze(-2)
            Y_I = (Q_f32 @ state_I_new).squeeze(-2).unsqueeze(2)
            Y_I = (self.inlier_norm(Y_I.float()) * self.gla_scale).to(dtype_in)

            Y_out = Y_I.transpose(1, 2).reshape(B, N, D)
            return self.out_proj(Y_out), None, state_I_new

        if self.r == self.d_head:
            curr_len = 1 if cache_O is None else cache_O[0].shape[2] + 1
            if getattr(self, 'rotary_emb', None) is not None:
                cos, sin = self.rotary_emb(curr_len)
                cos = cos[-1:]
                sin = sin[-1:]
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
                
            if cache_O is None:
                K_past, V_past = K, V
            else:
                K_past = torch.cat([cache_O[0], K], dim=2)
                V_past = torch.cat([cache_O[1], V], dim=2)
            Y = F.scaled_dot_product_attention(Q, K_past, V_past, is_causal=False, scale=1.0)
            cache_new = (K_past, V_past)
            Y_out = Y.transpose(1, 2).reshape(B, N, D)
            return self.out_proj(Y_out), cache_new, None

        gate_logits, mix_g = self._compute_gates_optimized(Q, K)

        # ----- outlier exact attention -----
        Q_O = Q[..., :self.r]
        K_O = K[..., :self.r]

        curr_len = 1 if cache_O is None else cache_O[0].shape[2] + 1
        if getattr(self, 'rotary_emb', None) is not None:
            cos, sin = self.rotary_emb(curr_len)
            cos = cos[-1:]
            sin = sin[-1:]
            Q_O, K_O = apply_rotary_pos_emb(Q_O, K_O, cos, sin)

        if cache_O is None:
            cache_O = (K_O, V)
        else:
            K_cache, V_cache = cache_O
            K_cache = torch.cat([K_cache, K_O], dim=2)
            V_cache = torch.cat([V_cache, V], dim=2)
            cache_O = (K_cache, V_cache)

        attn_weights = Q_O @ cache_O[0].transpose(-1, -2)
        sm_scale = (self.d_head / self.r) ** 0.5
        Y_O = torch.softmax(attn_weights * sm_scale, dim=-1) @ cache_O[1]

        # ----- inlier gated linear attention -----
        Q_J = Q[..., self.r:]
        K_J = K[..., self.r:]

        Q_J = F.pad(Q_J, (0, self.r))
        K_J = F.pad(K_J, (0, self.r))

        gamma = torch.sigmoid(-gate_logits).view(B, self.num_heads, 1, 1)

        if state_I is None:
            state_I = torch.zeros(B, self.num_heads, self.d_head, self.d_head,
                                  device=x.device, dtype=torch.float32)

        K_J_s = K_J.squeeze(2)
        V_s = V.squeeze(2)

        K_f32 = K_J_s.to(torch.float32).unsqueeze(-1)
        V_f32 = V_s.to(torch.float32).unsqueeze(-2)
        state_I = state_I * gamma.to(torch.float32) + (K_f32 @ V_f32)
        
        Q_f32 = Q_J.squeeze(2).to(torch.float32).unsqueeze(-2)
        Y_I = (Q_f32 @ state_I).squeeze(-2).unsqueeze(2)
        Y_I = (self.inlier_norm(Y_I.float()) * self.gla_scale).to(dtype_in)

        Y_out = (mix_g * Y_O) + ((1.0 - mix_g) * Y_I)
        Y_out = Y_out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(Y_out), cache_O, state_I


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
        
        std = 0.02
        res_std = std / math.sqrt(2 * model_cfg.num_layers)
        nn.init.normal_(self.mlp[0].weight, std=std)
        if self.mlp[0].bias is not None:
            nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=res_std)
        if self.mlp[2].bias is not None:
            nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


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
        
        nn.init.normal_(self.lm_head.weight, std=0.02)
        
        self.token_emb.weight = self.lm_head.weight

    def forward(self, x, targets=None):
        B, N = x.shape
        x = self.token_emb(x)
        for layer in self.layers:
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
