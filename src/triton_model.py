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
def hybrid_inlier_fused_kernel(
    Q_O_ptr, K_O_ptr, Q_J_ptr, K_J_ptr, V_ptr, Y_O_ptr,
    omega_O_ptr,
    Y_out_ptr,
    U, inv_sqrt_m_O: tl.constexpr,
    C: tl.constexpr, r_padded: tl.constexpr,
    j_padded: tl.constexpr, m_O: tl.constexpr,
    stride_qo_b, stride_qo_h, stride_qo_n, stride_qo_r,
    stride_ko_b, stride_ko_h, stride_ko_n, stride_ko_r,
    stride_qj_b, stride_qj_h, stride_qj_n, stride_qj_j,
    stride_kj_b, stride_kj_h, stride_kj_n, stride_kj_j,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_yo_b, stride_yo_h, stride_yo_n, stride_yo_d,
    stride_yout_b, stride_yout_h, stride_yout_n, stride_yout_d,
    stride_om_r, stride_om_m
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)  # Parallelized over d_head!

    # --- Base Pointers for current Batch, Head, and Feature Dim ---
    qo_base = Q_O_ptr + pid_b * stride_qo_b + pid_h * stride_qo_h
    ko_base = K_O_ptr + pid_b * stride_ko_b + pid_h * stride_ko_h
    qj_base = Q_J_ptr + pid_b * stride_qj_b + pid_h * stride_qj_h
    kj_base = K_J_ptr + pid_b * stride_kj_b + pid_h * stride_kj_h
    
    v_base = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h + pid_d * stride_v_d
    yo_base = Y_O_ptr + pid_b * stride_yo_b + pid_h * stride_yo_h + pid_d * stride_yo_d
    yout_base = Y_out_ptr + pid_b * stride_yout_b + pid_h * stride_yout_h + pid_d * stride_yout_d

    # --- Offsets ---
    offs_c = tl.arange(0, C)
    offs_r = tl.arange(0, r_padded)
    offs_m = tl.arange(0, m_O)
    offs_j = tl.arange(0, j_padded)

    # --- Initialize Recurrent States persistently in SRAM ---
    # These NEVER touch global memory during the sequence loop!
    S_phi = tl.zeros([m_O], dtype=tl.float32)
    S_K = tl.zeros([m_O, j_padded], dtype=tl.float32)
    S_KV_d = tl.zeros([m_O, j_padded], dtype=tl.float32)

    # Pre-load omega_O
    omega_O = tl.load(omega_O_ptr + offs_r[:, None] * stride_om_r + offs_m[None, :] * stride_om_m).to(tl.float32)

    # =====================================================================
    # FUSED SEQUENCE LOOP
    # =====================================================================
    for u in range(U):
        # Dynamic pointers for chunk u
        qo_ptr = qo_base + u * C * stride_qo_n
        ko_ptr = ko_base + u * C * stride_ko_n
        qj_ptr = qj_base + u * C * stride_qj_n
        kj_ptr = kj_base + u * C * stride_kj_n
        
        v_ptr_c = v_base + u * C * stride_v_n
        yo_ptr_c = yo_base + u * C * stride_yo_n
        yout_ptr_c = yout_base + u * C * stride_yout_n

        # --- Load Chunk Data ---
        Q_O = tl.load(qo_ptr + offs_c[:, None] * stride_qo_n + offs_r[None, :] * stride_qo_r)
        K_O = tl.load(ko_ptr + offs_c[:, None] * stride_ko_n + offs_r[None, :] * stride_ko_r)
        Q_J = tl.load(qj_ptr + offs_c[:, None] * stride_qj_n + offs_j[None, :] * stride_qj_j)
        K_J = tl.load(kj_ptr + offs_c[:, None] * stride_kj_n + offs_j[None, :] * stride_kj_j)
        
        V_d = tl.load(v_ptr_c + offs_c * stride_v_n)
        Y_O_d = tl.load(yo_ptr_c + offs_c * stride_yo_n)

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

        # Compile output and store chunk slice to HBM
        y_out_d = Y_O_d.to(tl.float32) * (1.0 - e) + term1_local_d + term1_past_d
        tl.store(yout_ptr_c + offs_c * stride_yout_n, y_out_d.to(Y_out_ptr.dtype.element_ty))

        # --- In-Place SRAM State Updates ---
        delta_S_KV_d = tl.dot(tl.trans(phi_K), K_J * V_d[:, None])                  
        S_KV_d += delta_S_KV_d.to(tl.float32)
        
        S_phi += tl.sum(phi_K.to(tl.float32), axis=0)
        
        delta_S_K = tl.dot(tl.trans(phi_K), K_J)                                    
        S_K += delta_S_K.to(tl.float32)


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

        # 1. Pad dimensions to minimum 16 for triton limits
        r_padded = max(16, int(2 ** math.ceil(math.log2(self.r)))) if self.r > 0 else 16
        j_padded = max(16, int(2 ** math.ceil(math.log2(self.j)))) if self.j > 0 else 16

        Q_O = F.pad(Q_O, (0, r_padded - self.r))
        K_O = F.pad(K_O, (0, r_padded - self.r))
        omega_O_padded = F.pad(self.omega_O, (0, 0, 0, r_padded - self.r)).contiguous()

        Q_J = F.pad(Q_J, (0, j_padded - self.j))
        K_J = F.pad(K_J, (0, j_padded - self.j))

        # 2. Global exact outlier attention via FlashAttention
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
        m_O = self.m_O
        inv_sqrt_m_O = float(1.0 / math.sqrt(m_O))

        Y_out = torch.empty_like(Y_O).contiguous()

        # 4. Fire Single Fused Kernel (Grid parallelized across feature dimension)
        grid = (B, self.num_heads, self.d_head)
        
        hybrid_inlier_fused_kernel[grid](
            Q_O, K_O, Q_J, K_J, V, Y_O,
            omega_O_padded,
            Y_out,
            U, inv_sqrt_m_O,
            C=C, r_padded=r_padded, j_padded=j_padded, m_O=m_O,
            stride_qo_b=Q_O.stride(0), stride_qo_h=Q_O.stride(1), stride_qo_n=Q_O.stride(2), stride_qo_r=Q_O.stride(3),
            stride_ko_b=K_O.stride(0), stride_ko_h=K_O.stride(1), stride_ko_n=K_O.stride(2), stride_ko_r=K_O.stride(3),
            stride_qj_b=Q_J.stride(0), stride_qj_h=Q_J.stride(1), stride_qj_n=Q_J.stride(2), stride_qj_j=Q_J.stride(3),
            stride_kj_b=K_J.stride(0), stride_kj_h=K_J.stride(1), stride_kj_n=K_J.stride(2), stride_kj_j=K_J.stride(3),
            stride_v_b=V.stride(0),   stride_v_h=V.stride(1),   stride_v_n=V.stride(2),   stride_v_d=V.stride(3),
            stride_yo_b=Y_O.stride(0), stride_yo_h=Y_O.stride(1), stride_yo_n=Y_O.stride(2), stride_yo_d=Y_O.stride(3),
            stride_yout_b=Y_out.stride(0), stride_yout_h=Y_out.stride(1), stride_yout_n=Y_out.stride(2), stride_yout_d=Y_out.stride(3),
            stride_om_r=omega_O_padded.stride(0), stride_om_m=omega_O_padded.stride(1), 
            num_warps=4, num_stages=2
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