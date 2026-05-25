import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
import triton
import triton.language as tl
from src.config import ModelConfig

# ---------- Triton Kernel ----------

@triton.jit
def hybrid_inlier_kernel(
    Q_O_ptr, K_O_ptr, Q_J_ptr, K_J_ptr, V_ptr, Y_O_ptr,
    omega_O_ptr,
    S_phi_ptr, S_K_ptr, S_KV_ptr, S_V_ptr,
    Y_out_ptr,
    inv_sqrt_m_O: tl.constexpr,
    C: tl.constexpr, r_padded: tl.constexpr,
    j_padded: tl.constexpr, d_head, m_O: tl.constexpr,
    stride_qo_b, stride_qo_h, stride_qo_c, stride_qo_r,
    stride_ko_b, stride_ko_h, stride_ko_c, stride_ko_r,
    stride_qj_b, stride_qj_h, stride_qj_c, stride_qj_j,
    stride_kj_b, stride_kj_h, stride_kj_c, stride_kj_j,
    stride_v_b, stride_v_h, stride_v_c, stride_v_d,
    stride_yo_b, stride_yo_h, stride_yo_c, stride_yo_d,
    stride_sphi_b, stride_sphi_h, stride_sphi_m,
    stride_sk_b, stride_sk_h, stride_sk_m, stride_sk_j,
    stride_skv_b, stride_skv_h, stride_skv_m, stride_skv_j, stride_skv_d,
    stride_sv_b, stride_sv_h, stride_sv_m, stride_sv_d,
    stride_yout_b, stride_yout_h, stride_yout_c, stride_yout_d,
    stride_om_r, stride_om_m
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # --- Pointers for current Batch and Head ---
    qo_ptr = Q_O_ptr + pid_b * stride_qo_b + pid_h * stride_qo_h
    ko_ptr = K_O_ptr + pid_b * stride_ko_b + pid_h * stride_ko_h
    qj_ptr = Q_J_ptr + pid_b * stride_qj_b + pid_h * stride_qj_h
    kj_ptr = K_J_ptr + pid_b * stride_kj_b + pid_h * stride_kj_h
    v_ptr  = V_ptr   + pid_b * stride_v_b  + pid_h * stride_v_h
    yo_ptr = Y_O_ptr + pid_b * stride_yo_b + pid_h * stride_yo_h
    yout_ptr = Y_out_ptr + pid_b * stride_yout_b + pid_h * stride_yout_h
    
    sphi_ptr = S_phi_ptr + pid_b * stride_sphi_b + pid_h * stride_sphi_h
    sk_ptr   = S_K_ptr   + pid_b * stride_sk_b   + pid_h * stride_sk_h
    skv_ptr  = S_KV_ptr  + pid_b * stride_skv_b  + pid_h * stride_skv_h
    sv_ptr   = S_V_ptr   + pid_b * stride_sv_b   + pid_h * stride_sv_h

    # --- Offsets ---
    offs_c = tl.arange(0, C)
    offs_r = tl.arange(0, r_padded)
    offs_m = tl.arange(0, m_O)
    offs_j = tl.arange(0, j_padded)

    # --- Load Chunk Data (Float32 for stability) ---
    Q_O = tl.load(qo_ptr + offs_c[:, None] * stride_qo_c + offs_r[None, :] * stride_qo_r).to(tl.float32)
    K_O = tl.load(ko_ptr + offs_c[:, None] * stride_ko_c + offs_r[None, :] * stride_ko_r).to(tl.float32)
    Q_J = tl.load(qj_ptr + offs_c[:, None] * stride_qj_c + offs_j[None, :] * stride_qj_j).to(tl.float32)
    K_J = tl.load(kj_ptr + offs_c[:, None] * stride_kj_c + offs_j[None, :] * stride_kj_j).to(tl.float32)
    omega_O = tl.load(omega_O_ptr + offs_r[:, None] * stride_om_r + offs_m[None, :] * stride_om_m).to(tl.float32)

    # --- Compute Phi (Random Fourier Features) ---
    Q_O_norm = tl.sum(Q_O * Q_O, axis=1)[:, None] * 0.5
    K_O_norm = tl.sum(K_O * K_O, axis=1)[:, None] * 0.5

    phi_Q_logit = tl.dot(Q_O, omega_O)
    phi_K_logit = tl.dot(K_O, omega_O)

    phi_Q = tl.exp(phi_Q_logit - Q_O_norm) * inv_sqrt_m_O
    phi_K = tl.exp(phi_K_logit - K_O_norm) * inv_sqrt_m_O

    # --- Local Exact Inlier Correction ---
    logits_local = tl.dot(Q_O, tl.trans(K_O))
    mask_causal = offs_c[:, None] >= offs_c[None, :]
    logits_local = tl.where(mask_causal, logits_local, -float('inf'))

    l_max = tl.max(logits_local, axis=1)
    l_exp = tl.exp(logits_local - l_max[:, None])
    l_sum = tl.sum(l_exp, axis=1)
    P_local = l_exp / l_sum[:, None]

    weighted_K_J = tl.dot(P_local, K_J)                # (C, j_padded)
    e_local = tl.sum(Q_J * weighted_K_J, axis=1)       # (C,)

    # Pre-compute combined coefficient matrix for local terms
    E = tl.dot(Q_J, tl.trans(K_J))                     # (C, C)
    P_local_E = P_local * E                            # (C, C)

    # --- Past Correction via Recurrent States ---
    S_phi = tl.load(sphi_ptr + offs_m * stride_sphi_m) # (m_O,)
    Z_past = tl.sum(phi_Q * S_phi[None, :], axis=1)    # (C,)

    attn_phi_full = tl.dot(phi_Q, tl.trans(phi_K))     # (C, C)
    attn_phi_causal = tl.where(mask_causal, attn_phi_full, 0.0)
    Z_local = tl.sum(attn_phi_causal, axis=1)
    Z_total = Z_past + Z_local + 1e-8

    # e_past
    S_K = tl.load(sk_ptr + offs_m[:, None] * stride_sk_m + offs_j[None, :] * stride_sk_j)
    W_K = tl.dot(phi_Q, S_K)                                    # (C, j_padded)
    e_past = tl.sum(Q_J * W_K, axis=1) / Z_total                # (C,)
    e = e_local + e_past

    # --- Output and S_KV Update (Loop over d_head dynamically) ---
    # range() creates a clean dynamic loop avoiding instruction cache explosion.
    for d_idx in range(d_head):
        # Dynamically load the exact required slices
        S_KV_d = tl.load(skv_ptr + offs_m[:, None] * stride_skv_m + offs_j[None, :] * stride_skv_j + d_idx * stride_skv_d)
        V_d = tl.load(v_ptr + offs_c * stride_v_c + d_idx * stride_v_d)       # (C,)
        Y_O_d = tl.load(yo_ptr + offs_c * stride_yo_c + d_idx * stride_yo_d)  # (C,)
        S_V_d = tl.load(sv_ptr + offs_m * stride_sv_m + d_idx * stride_sv_d)  # (m_O,)

        # 1. Past projection logic
        W_KV_d = tl.dot(phi_Q, S_KV_d)                                        # (C, j_padded)
        term1_past_d = tl.sum(Q_J * W_KV_d, axis=1) / Z_total                 # (C,)

        # 2. Local projection logic (calculated dynamically, avoiding huge block allocation)
        term1_local_d = tl.sum(P_local_E * V_d[None, :], axis=1)              # (C,)

        # 3. Compile output
        y_out_d = Y_O_d * (1.0 - e) + term1_local_d + term1_past_d            # (C,)
        tl.store(yout_ptr + offs_c * stride_yout_c + d_idx * stride_yout_d, y_out_d.to(Y_out_ptr.dtype.element_ty))

        # 4. Update S_KV_d state
        delta_S_KV_d = tl.dot(tl.trans(phi_K), K_J * V_d[:, None])            # (m_O, j_padded)
        tl.store(skv_ptr + offs_m[:, None] * stride_skv_m + offs_j[None, :] * stride_skv_j + d_idx * stride_skv_d, S_KV_d + delta_S_KV_d)
        
        # 5. Update S_V_d state inside the same loop to reuse V_d!
        delta_S_V_d = tl.sum(phi_K * V_d[:, None], axis=0)                    # (m_O,)
        tl.store(sv_ptr + offs_m * stride_sv_m + d_idx * stride_sv_d, S_V_d + delta_S_V_d)

    # --- Final State Updates (S_phi, S_K) ---
    tl.store(sphi_ptr + offs_m * stride_sphi_m, S_phi + tl.sum(phi_K, axis=0))
    tl.store(sk_ptr + offs_m[:, None] * stride_sk_m + offs_j[None, :] * stride_sk_j, S_K + tl.dot(tl.trans(phi_K), K_J))


# ---------- Optimized Module ----------

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
        self.chunk_size = model_cfg.chunk_size

        self.register_buffer('omega_J', torch.randn(self.j, self.m))
        self.register_buffer('omega_O', torch.randn(self.r, self.m_O))

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)

        self.register_buffer('_cached_outlier_idx', None)
        self.register_buffer('_cached_inlier_idx', None)
        self._step_counter = 0
        self._refresh_steps = 100

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

        Q = (self.W_q(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        K = (self.W_k(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)

        # ---- Special case: r == 0 -> pure causal linear attention ----
        if self.r == 0:
            phi_Q = torch.exp(Q @ self.omega_J.to(dtype_in) - (Q ** 2).sum(dim=-1, keepdim=True) / 2.0) / math.sqrt(self.m)
            phi_K = torch.exp(K @ self.omega_J.to(dtype_in) - (K ** 2).sum(dim=-1, keepdim=True) / 2.0) / math.sqrt(self.m)
            KV = torch.einsum('bhnm, bhnd -> bhnmd', phi_K, V)
            KV_cum = torch.cumsum(KV, dim=2)
            K_cum = torch.cumsum(phi_K, dim=2)
            num = torch.einsum('bhnm, bhnmd -> bhnd', phi_Q, KV_cum)
            den = torch.einsum('bhnm, bhnm -> bhn', phi_Q, K_cum).unsqueeze(-1)
            Y = num / (den + 1e-8)
            return Y.transpose(1, 2).reshape(B, N, D)

        # ---- Special case: r == d_head -> true exact softmax ----
        if self.r == self.d_head:
            attn = torch.einsum('bhqd,bhkd->bhqk', Q, K)
            causal_mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
            attn = torch.where(causal_mask, attn, torch.tensor(torch.finfo(attn.dtype).min, device=x.device, dtype=attn.dtype))
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

        # 1. Pad dimensions to minimum 16 for triton limits, guarantees fully aligned math
        r_padded = max(16, int(2 ** math.ceil(math.log2(self.r)))) if self.r > 0 else 16
        j_padded = max(16, int(2 ** math.ceil(math.log2(self.j)))) if self.j > 0 else 16

        Q_O = F.pad(Q_O, (0, r_padded - self.r))
        K_O = F.pad(K_O, (0, r_padded - self.r))
        omega_O_padded = F.pad(self.omega_O, (0, 0, 0, r_padded - self.r)).contiguous()

        Q_J = F.pad(Q_J, (0, j_padded - self.j))
        K_J = F.pad(K_J, (0, j_padded - self.j))

        # 2. Global exact outlier attention via FlashAttention (scale=1.0 maps accurately to unoptimized setup)
        Y_O = F.scaled_dot_product_attention(Q_O, K_O, V, is_causal=True, scale=1.0)

        # 3. Pad sequence length to multiple of chunk_size
        C = self.chunk_size
        pad_len = (C - (N % C)) % C
        if pad_len > 0:
            Q_O = F.pad(Q_O, (0, 0, 0, pad_len))
            K_O = F.pad(K_O, (0, 0, 0, pad_len))
            Q_J = F.pad(Q_J, (0, 0, 0, pad_len))
            K_J = F.pad(K_J, (0, 0, 0, pad_len))
            V   = F.pad(V,   (0, 0, 0, pad_len))
            Y_O = F.pad(Y_O, (0, 0, 0, pad_len))

        U = (N + pad_len) // C

        # 4. Initialize recurrent states
        m_O = self.m_O
        S_phi = torch.zeros(B, self.num_heads, m_O, device=x.device, dtype=torch.float32).contiguous()
        S_K   = torch.zeros(B, self.num_heads, m_O, j_padded, device=x.device, dtype=torch.float32).contiguous()
        S_KV  = torch.zeros(B, self.num_heads, m_O, j_padded, self.d_head, device=x.device, dtype=torch.float32).contiguous()
        S_V   = torch.zeros(B, self.num_heads, m_O, self.d_head, device=x.device, dtype=torch.float32).contiguous()

        Y_out = torch.empty_like(Y_O).contiguous()
        grid = (B, self.num_heads)

        inv_sqrt_m_O = 1.0 / math.sqrt(m_O)

        for u in range(U):
            # Pass slice views (stride correctly propagates). No .contiguous() here to avoid copy overhead/write-back loss!
            q_o_c = Q_O[:, :, u*C:(u+1)*C, :]
            k_o_c = K_O[:, :, u*C:(u+1)*C, :]
            q_j_c = Q_J[:, :, u*C:(u+1)*C, :]
            k_j_c = K_J[:, :, u*C:(u+1)*C, :]
            v_c   = V[:, :, u*C:(u+1)*C, :]
            y_o_c = Y_O[:, :, u*C:(u+1)*C, :]
            y_out_c = Y_out[:, :, u*C:(u+1)*C, :]

            hybrid_inlier_kernel[grid](
                q_o_c, k_o_c, q_j_c, k_j_c, v_c, y_o_c,
                omega_O_padded,
                S_phi, S_K, S_KV, S_V,
                y_out_c,
                inv_sqrt_m_O,
                C=C, r_padded=r_padded,
                j_padded=j_padded, d_head=self.d_head, m_O=m_O,
                stride_qo_b=q_o_c.stride(0), stride_qo_h=q_o_c.stride(1),
                stride_qo_c=q_o_c.stride(2), stride_qo_r=q_o_c.stride(3),
                stride_ko_b=k_o_c.stride(0), stride_ko_h=k_o_c.stride(1),
                stride_ko_c=k_o_c.stride(2), stride_ko_r=k_o_c.stride(3),
                stride_qj_b=q_j_c.stride(0), stride_qj_h=q_j_c.stride(1),
                stride_qj_c=q_j_c.stride(2), stride_qj_j=q_j_c.stride(3),
                stride_kj_b=k_j_c.stride(0), stride_kj_h=k_j_c.stride(1),
                stride_kj_c=k_j_c.stride(2), stride_kj_j=k_j_c.stride(3),
                stride_v_b=v_c.stride(0), stride_v_h=v_c.stride(1),
                stride_v_c=v_c.stride(2), stride_v_d=v_c.stride(3),
                stride_yo_b=y_o_c.stride(0), stride_yo_h=y_o_c.stride(1),
                stride_yo_c=y_o_c.stride(2), stride_yo_d=y_o_c.stride(3),
                stride_sphi_b=S_phi.stride(0), stride_sphi_h=S_phi.stride(1),
                stride_sphi_m=S_phi.stride(2),
                stride_sk_b=S_K.stride(0), stride_sk_h=S_K.stride(1),
                stride_sk_m=S_K.stride(2), stride_sk_j=S_K.stride(3),
                stride_skv_b=S_KV.stride(0), stride_skv_h=S_KV.stride(1),
                stride_skv_m=S_KV.stride(2), stride_skv_j=S_KV.stride(3), stride_skv_d=S_KV.stride(4),
                stride_sv_b=S_V.stride(0), stride_sv_h=S_V.stride(1),
                stride_sv_m=S_V.stride(2), stride_sv_d=S_V.stride(3),
                stride_yout_b=y_out_c.stride(0), stride_yout_h=y_out_c.stride(1),
                stride_yout_c=y_out_c.stride(2), stride_yout_d=y_out_c.stride(3),
                stride_om_r=omega_O_padded.stride(0), stride_om_m=omega_O_padded.stride(1), 
                num_warps=4,      # default is usually 8
                num_stages=2,      # reduce pipeline stages
            )

        if pad_len > 0:
            Y_out = Y_out[:, :, :-pad_len]

        return Y_out.transpose(1, 2).reshape(B, N, D).to(dtype_in)


# ---------- Full Model ----------

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
