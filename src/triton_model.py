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
def hybrid_single_chunk_kernel(
    Q_O_ptr, K_O_ptr, Q_ptr, K_ptr, V_ptr, Y_O_ptr,
    omega_O_ptr, Y_out_ptr,
    S_phi_ptr, S_K_ptr, S_KV_ptr,
    inlier_idx_ptr,
    u, N, r, j, inv_sqrt_m_O: tl.constexpr, 
    C: tl.constexpr, r_padded: tl.constexpr,
    j_padded: tl.constexpr, m_O: tl.constexpr,
    stride_qo_b, stride_qo_h, stride_qo_n, stride_qo_r,
    stride_ko_b, stride_ko_h, stride_ko_n, stride_ko_r,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_yo_b, stride_yo_h, stride_yo_n, stride_yo_d,
    stride_yout_b, stride_yout_h, stride_yout_n, stride_yout_d,
    stride_om_r, stride_om_m,
    stride_sphi_b, stride_sphi_h, stride_sphi_m,
    stride_sk_b, stride_sk_h, stride_sk_m, stride_sk_j,
    stride_skv_b, stride_skv_h, stride_skv_d, stride_skv_m, stride_skv_j,
    stride_idx_h
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)

    # --- Offsets & Masks ---
    offs_c = tl.arange(0, C)
    offs_r = tl.arange(0, r_padded)
    offs_m = tl.arange(0, m_O)
    offs_j = tl.arange(0, j_padded)

    # Sequence bounds masking (No PyTorch padding needed!)
    n_idx = u * C + offs_c
    mask_n = n_idx < N
    mask_r = offs_r < r
    mask_j = offs_j < j

    # --- Load Inlier Indices ---
    inlier_indices = tl.load(inlier_idx_ptr + pid_h * stride_idx_h + offs_j, mask=mask_j, other=0)

    # --- Base Pointers ---
    v_ptrs = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h + n_idx * stride_v_n + pid_d * stride_v_d
    yo_ptrs = Y_O_ptr + pid_b * stride_yo_b + pid_h * stride_yo_h + n_idx * stride_yo_n + pid_d * stride_yo_d
    yout_ptrs = Y_out_ptr + pid_b * stride_yout_b + pid_h * stride_yout_h + n_idx * stride_yout_n + pid_d * stride_yout_d

    sphi_base = S_phi_ptr + pid_b * stride_sphi_b + pid_h * stride_sphi_h
    sk_base = S_K_ptr + pid_b * stride_sk_b + pid_h * stride_sk_h
    skv_base = S_KV_ptr + pid_b * stride_skv_b + pid_h * stride_skv_h + pid_d * stride_skv_d

    ptrs_sphi = sphi_base + offs_m * stride_sphi_m
    ptrs_sk = sk_base + offs_m[:, None] * stride_sk_m + offs_j[None, :] * stride_sk_j
    ptrs_skv = skv_base + offs_m[:, None] * stride_skv_m + offs_j[None, :] * stride_skv_j

    S_phi = tl.load(ptrs_sphi)
    S_K = tl.load(ptrs_sk)
    S_KV_d = tl.load(ptrs_skv)

    omega_O = tl.load(omega_O_ptr + offs_r[:, None] * stride_om_r + offs_m[None, :] * stride_om_m, mask=mask_r[:, None], other=0.0).to(tl.float32)

    # --- Load Chunk Data (Masked & Gathered in SRAM) ---
    # Q_O is loaded from the pre-sliced outlier tensor
    qo_ptrs = Q_O_ptr + pid_b * stride_qo_b + pid_h * stride_qo_h + n_idx[:, None] * stride_qo_n + offs_r[None, :] * stride_qo_r
    ko_ptrs = K_O_ptr + pid_b * stride_ko_b + pid_h * stride_ko_h + n_idx[:, None] * stride_ko_n + offs_r[None, :] * stride_ko_r
    
    Q_O = tl.load(qo_ptrs, mask=mask_n[:, None] & mask_r[None, :], other=0.0)
    K_O = tl.load(ko_ptrs, mask=mask_n[:, None] & mask_r[None, :], other=0.0)

    # Q_J is GATHERED directly from the raw Q tensor!
    qj_ptrs = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h + n_idx[:, None] * stride_q_n + inlier_indices[None, :] * stride_q_d
    kj_ptrs = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h + n_idx[:, None] * stride_k_n + inlier_indices[None, :] * stride_k_d
    
    Q_J = tl.load(qj_ptrs, mask=mask_n[:, None] & mask_j[None, :], other=0.0)
    K_J = tl.load(kj_ptrs, mask=mask_n[:, None] & mask_j[None, :], other=0.0)
    
    V_d = tl.load(v_ptrs, mask=mask_n, other=0.0)
    Y_O_d = tl.load(yo_ptrs, mask=mask_n, other=0.0)

    native_dtype = Q_O.dtype

    # --- Compute Phi ---
    Q_O_f32 = Q_O.to(tl.float32)
    K_O_f32 = K_O.to(tl.float32)

    Q_O_norm = tl.sum(Q_O_f32 * Q_O_f32, axis=1)[:, None] * 0.5
    K_O_norm = tl.sum(K_O_f32 * K_O_f32, axis=1)[:, None] * 0.5

    phi_Q_logit = tl.dot(Q_O, omega_O.to(native_dtype))
    phi_K_logit = tl.dot(K_O, omega_O.to(native_dtype))

    phi_Q = (tl.exp(phi_Q_logit.to(tl.float32) - Q_O_norm) * inv_sqrt_m_O).to(native_dtype)
    phi_K = (tl.exp(phi_K_logit.to(tl.float32) - K_O_norm) * inv_sqrt_m_O).to(native_dtype)

    # --- Local Exact Inlier Correction ---
    logits_local = tl.dot(Q_O, tl.trans(K_O))
    mask_causal = offs_c[:, None] >= offs_c[None, :]
    logits_local = tl.where(mask_causal, logits_local.to(tl.float32), -float('inf'))

    l_max = tl.max(logits_local, axis=1)
    l_exp = tl.exp(logits_local - l_max[:, None])
    l_sum = tl.sum(l_exp, axis=1)
    P_local_f32 = l_exp / l_sum[:, None]
    
    P_local = P_local_f32.to(native_dtype)
    weighted_K_J = tl.dot(P_local, K_J)                                         
    e_local = tl.sum(Q_J.to(tl.float32) * weighted_K_J.to(tl.float32), axis=1)  

    E = tl.dot(Q_J, tl.trans(K_J))                                              
    P_local_E = (P_local_f32 * E.to(tl.float32)).to(native_dtype)               

    # --- Past Correction via Recurrent States ---
    Z_past = tl.sum(phi_Q.to(tl.float32) * S_phi[None, :], axis=1)              
    attn_phi_full = tl.dot(phi_Q, tl.trans(phi_K))                              
    attn_phi_causal = tl.where(mask_causal, attn_phi_full.to(tl.float32), 0.0)
    Z_local = tl.sum(attn_phi_causal, axis=1)
    Z_total = Z_past + Z_local + 1e-8

    # e_past
    W_K = tl.dot(phi_Q, S_K.to(native_dtype))                                   
    e_past = tl.sum(Q_J.to(tl.float32) * W_K.to(tl.float32), axis=1) / Z_total  
    e = e_local + e_past

    # --- Output Calculations ---
    W_KV_d = tl.dot(phi_Q, S_KV_d.to(native_dtype))                             
    term1_past_d = tl.sum(Q_J.to(tl.float32) * W_KV_d.to(tl.float32), axis=1) / Z_total  
    term1_local_d = tl.sum(P_local_E.to(tl.float32) * V_d.to(tl.float32)[None, :], axis=1) 

    # Masked Output Write
    y_out_d = Y_O_d.to(tl.float32) * (1.0 - e) + term1_local_d + term1_past_d
    tl.store(yout_ptrs, y_out_d.to(Y_out_ptr.dtype.element_ty), mask=mask_n)

    # --- State Updates ---
    # Apply mask_n so padded sequence garbage doesn't corrupt our global state!
    mask_n_f32 = tl.where(mask_n, 1.0, 0.0)[:, None]
    phi_K_masked = phi_K.to(tl.float32) * mask_n_f32
    K_J_masked = K_J.to(tl.float32) * mask_n_f32

    S_KV_d += tl.dot(tl.trans(phi_K_masked), K_J_masked * V_d[:, None])
    S_phi += tl.sum(phi_K_masked, axis=0)
    S_K += tl.dot(tl.trans(phi_K_masked), K_J_masked)

    # --- Write Back to Global Memory ---
    tl.store(ptrs_skv, S_KV_d)
    if pid_d == 0:
        tl.store(ptrs_sphi, S_phi)
        tl.store(ptrs_sk, S_K)


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

        Q = (self.W_q(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2).contiguous()
        K = (self.W_k(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2).contiguous()
        V = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2).contiguous()

        # [Special cases for r==0 and r==d_head remain unchanged...]

        self._maybe_update_indices()
        outlier_idx = self._cached_outlier_idx
        inlier_idx = self._cached_inlier_idx

        # 1. We ONLY gather the outliers for the PyTorch SDPA pass. No padding!
        out_gather = outlier_idx.view(1, self.num_heads, 1, self.r).expand(B, self.num_heads, N, self.r)
        Q_O = Q.gather(-1, out_gather)
        K_O = K.gather(-1, out_gather)

        # 2. Global exact outlier attention (Still O(N^2) but with vastly smaller constant)
        Y_O = F.scaled_dot_product_attention(Q_O, K_O, V, is_causal=True, scale=1.0)
        Y_out = torch.empty_like(Y_O).contiguous()

        # 3. Prepare Triton constants
        r_padded = max(16, int(2 ** math.ceil(math.log2(self.r)))) if self.r > 0 else 16
        j_padded = max(16, int(2 ** math.ceil(math.log2(self.j)))) if self.j > 0 else 16
        C = self.chunk_size
        U = math.ceil(N / C)
        m_O = self.m_O
        inv_sqrt_m_O = float(1.0 / math.sqrt(m_O))

        # 4. Initialize Persistent Global States
        S_phi = torch.zeros((B, self.num_heads, m_O), device=x.device, dtype=torch.float32)
        S_K = torch.zeros((B, self.num_heads, m_O, j_padded), device=x.device, dtype=torch.float32)
        S_KV = torch.zeros((B, self.num_heads, self.d_head, m_O, j_padded), device=x.device, dtype=torch.float32)

        # 5. Fire Sequential Chunks
        grid = (B, self.num_heads, self.d_head)
        
        for u in range(U):
            hybrid_single_chunk_kernel[grid](
                Q_O, K_O, Q, K, V, Y_O,
                self.omega_O, Y_out,
                S_phi, S_K, S_KV,
                inlier_idx,
                u, N, self.r, self.j, inv_sqrt_m_O,
                C=C, r_padded=r_padded, j_padded=j_padded, m_O=m_O,
                stride_qo_b=Q_O.stride(0), stride_qo_h=Q_O.stride(1), stride_qo_n=Q_O.stride(2), stride_qo_r=Q_O.stride(3),
                stride_ko_b=K_O.stride(0), stride_ko_h=K_O.stride(1), stride_ko_n=K_O.stride(2), stride_ko_r=K_O.stride(3),
                stride_q_b=Q.stride(0), stride_q_h=Q.stride(1), stride_q_n=Q.stride(2), stride_q_d=Q.stride(3),
                stride_k_b=K.stride(0), stride_k_h=K.stride(1), stride_k_n=K.stride(2), stride_k_d=K.stride(3),
                stride_v_b=V.stride(0), stride_v_h=V.stride(1), stride_v_n=V.stride(2), stride_v_d=V.stride(3),
                stride_yo_b=Y_O.stride(0), stride_yo_h=Y_O.stride(1), stride_yo_n=Y_O.stride(2), stride_yo_d=Y_O.stride(3),
                stride_yout_b=Y_out.stride(0), stride_yout_h=Y_out.stride(1), stride_yout_n=Y_out.stride(2), stride_yout_d=Y_out.stride(3),
                stride_om_r=self.omega_O.stride(0), stride_om_m=self.omega_O.stride(1), 
                stride_sphi_b=S_phi.stride(0), stride_sphi_h=S_phi.stride(1), stride_sphi_m=S_phi.stride(2),
                stride_sk_b=S_K.stride(0), stride_sk_h=S_K.stride(1), stride_sk_m=S_K.stride(2), stride_sk_j=S_K.stride(3),
                stride_skv_b=S_KV.stride(0), stride_skv_h=S_KV.stride(1), stride_skv_d=S_KV.stride(2), stride_skv_m=S_KV.stride(3), stride_skv_j=S_KV.stride(4),
                stride_idx_h=inlier_idx.stride(0),
                num_warps=4, num_stages=2
            )

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