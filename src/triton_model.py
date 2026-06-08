import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl
from src.config import ModelConfig

@triton.jit
def hybrid_single_chunk_kernel(
    Q_ptr, K_ptr, V_ptr, Y_out_ptr,
    omega_O_ptr,
    S_phi_ptr, S_K_ptr, S_KV_ptr,
    outlier_idx_ptr, inlier_idx_ptr,
    u, N, r, j, inv_sqrt_m_O: tl.constexpr,
    C: tl.constexpr, r_padded: tl.constexpr,
    j_padded: tl.constexpr, m_O: tl.constexpr,
    inlier_scale: tl.constexpr,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_yout_b, stride_yout_h, stride_yout_n, stride_yout_d,
    stride_om_r, stride_om_m,
    stride_sphi_b, stride_sphi_h, stride_sphi_m,
    stride_sk_b, stride_sk_h, stride_sk_m, stride_sk_j,
    stride_skv_b, stride_skv_h, stride_skv_d, stride_skv_m, stride_skv_j,
    stride_oi_h, stride_oi_r,
    stride_ii_h, stride_ii_j
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_c = tl.arange(0, C)
    offs_r = tl.arange(0, r_padded)
    offs_m = tl.arange(0, m_O)
    offs_j = tl.arange(0, j_padded)

    n_idx = u * C + offs_c
    mask_n = n_idx < N
    mask_r = offs_r < r
    mask_j = offs_j < j

    outlier_idx = tl.load(outlier_idx_ptr + pid_h * stride_oi_h + offs_r * stride_oi_r,
                          mask=mask_r, other=0)
    inlier_idx  = tl.load(inlier_idx_ptr  + pid_h * stride_ii_h + offs_j * stride_ii_j,
                          mask=mask_j, other=0)

    v_ptrs     = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h + n_idx * stride_v_n + pid_d * stride_v_d
    yout_ptrs  = Y_out_ptr + pid_b * stride_yout_b + pid_h * stride_yout_h + n_idx * stride_yout_n + pid_d * stride_yout_d

    sphi_base = S_phi_ptr + pid_b * stride_sphi_b + pid_h * stride_sphi_h
    sk_base   = S_K_ptr   + pid_b * stride_sk_b   + pid_h * stride_sk_h
    skv_base  = S_KV_ptr  + pid_b * stride_skv_b  + pid_h * stride_skv_h + pid_d * stride_skv_d

    qo_ptrs = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h + n_idx[:, None] * stride_q_n + outlier_idx[None, :] * stride_q_d
    ko_ptrs = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h + n_idx[:, None] * stride_k_n + outlier_idx[None, :] * stride_k_d
    Q_O = tl.load(qo_ptrs, mask=mask_n[:, None] & mask_r[None, :], other=0.0)
    K_O = tl.load(ko_ptrs, mask=mask_n[:, None] & mask_r[None, :], other=0.0)

    qj_ptrs = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h + n_idx[:, None] * stride_q_n + inlier_idx[None, :] * stride_q_d
    kj_ptrs = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h + n_idx[:, None] * stride_k_n + inlier_idx[None, :] * stride_k_d
    Q_J = tl.load(qj_ptrs, mask=mask_n[:, None] & mask_j[None, :], other=0.0)
    K_J = tl.load(kj_ptrs, mask=mask_n[:, None] & mask_j[None, :], other=0.0)

    V_d    = tl.load(v_ptrs,    mask=mask_n, other=0.0)
    Y_O_d  = tl.load(yout_ptrs, mask=mask_n, other=0.0)
    omega_O = tl.load(omega_O_ptr + offs_r[:, None] * stride_om_r + offs_m[None, :] * stride_om_m,
                      mask=mask_r[:, None], other=0.0).to(tl.float32)

    native_dtype = Q_O.dtype

    # random Fourier features (phi)
    Q_O_f32 = Q_O.to(tl.float32)
    K_O_f32 = K_O.to(tl.float32)
    Q_O_norm = tl.sum(Q_O_f32 * Q_O_f32, axis=1)[:, None] * 0.5
    K_O_norm = tl.sum(K_O_f32 * K_O_f32, axis=1)[:, None] * 0.5

    phi_Q_logit = tl.dot(Q_O, omega_O.to(native_dtype))
    phi_K_logit = tl.dot(K_O, omega_O.to(native_dtype))
    phi_Q = (tl.exp(phi_Q_logit.to(tl.float32) - Q_O_norm) * inv_sqrt_m_O).to(native_dtype)
    phi_K = (tl.exp(phi_K_logit.to(tl.float32) - K_O_norm) * inv_sqrt_m_O).to(native_dtype)

    # local exact inlier correction
    logits_local = tl.dot(Q_O, tl.trans(K_O))
    mask_causal = offs_c[:, None] >= offs_c[None, :]
    logits_local = tl.where(mask_causal, logits_local.to(tl.float32), -float('inf'))

    l_max    = tl.max(logits_local, axis=1)
    l_exp    = tl.exp(logits_local - l_max[:, None])
    l_sum    = tl.sum(l_exp, axis=1)
    P_local_f32 = l_exp / l_sum[:, None]
    P_local     = P_local_f32.to(native_dtype)

    weighted_K_J = tl.dot(P_local, K_J)
    e_local = tl.sum(Q_J.to(tl.float32) * weighted_K_J.to(tl.float32), axis=1) / inlier_scale

    E = tl.dot(Q_J, tl.trans(K_J)) / inlier_scale
    P_local_E = (P_local_f32 * E.to(tl.float32)).to(native_dtype)

    # past correction via recurrent states
    ptrs_sphi = sphi_base + offs_m * stride_sphi_m
    ptrs_sk   = sk_base   + offs_m[:, None] * stride_sk_m + offs_j[None, :] * stride_sk_j
    ptrs_skv  = skv_base  + offs_m[:, None] * stride_skv_m + offs_j[None, :] * stride_skv_j

    S_phi  = tl.load(ptrs_sphi)
    S_K    = tl.load(ptrs_sk)
    S_KV_d = tl.load(ptrs_skv)

    Z_past = tl.sum(phi_Q.to(tl.float32) * S_phi[None, :], axis=1)
    attn_phi_full = tl.dot(phi_Q, tl.trans(phi_K))
    attn_phi_causal = tl.where(mask_causal, attn_phi_full.to(tl.float32), 0.0)
    Z_local = tl.sum(attn_phi_causal, axis=1)
    Z_total = Z_past + Z_local + 1e-8

    W_K = tl.dot(phi_Q, S_K.to(native_dtype))
    e_past = tl.sum(Q_J.to(tl.float32) * W_K.to(tl.float32), axis=1) / Z_total
    e = e_local + e_past
    e = tl.maximum(tl.minimum(e, 0.95), -0.95)

    W_KV_d = tl.dot(phi_Q, S_KV_d.to(native_dtype))
    term1_past_d  = tl.sum(Q_J.to(tl.float32) * W_KV_d.to(tl.float32), axis=1) / Z_total
    term1_local_d = tl.sum(P_local_E.to(tl.float32) * V_d.to(tl.float32)[None, :], axis=1)

    y_out_d = Y_O_d.to(tl.float32) * (1.0 - e) + term1_local_d + term1_past_d
    tl.store(yout_ptrs, y_out_d.to(Y_out_ptr.dtype.element_ty), mask=mask_n)

    # update recurrent states
    mask_n_f32 = tl.where(mask_n, 1.0, 0.0)[:, None]
    phi_K_masked = phi_K.to(tl.float32) * mask_n_f32
    K_J_masked   = K_J.to(tl.float32) * mask_n_f32
    K_J_scaled   = K_J_masked * (1.0 / inlier_scale)

    S_KV_d += tl.dot(tl.trans(phi_K_masked), K_J_scaled * V_d[:, None])
    S_phi  += tl.sum(phi_K_masked, axis=0)
    S_K    += tl.dot(tl.trans(phi_K_masked), K_J_scaled)

    tl.store(ptrs_skv, S_KV_d)
    if pid_d == 0:
        tl.store(ptrs_sphi, S_phi)
        tl.store(ptrs_sk,   S_K)


# -----------------------------------------------------------------------------
# Autograd wrapper (corrected kernel call and inlier masking)
# -----------------------------------------------------------------------------
class HOFAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q, K, V, Y_out_base,
        omega_O,
        outlier_idx, inlier_idx,
        C, r_padded, j_padded, m_O,
        inlier_scale, inv_sqrt_m_O,
        N, pad_len
    ):
        B, H, N_pad, d_head = Q.shape
        assert N_pad == (N + pad_len)
        U = N_pad // C

        # Clone Y_out to avoid modifying the input base tensor
        Y_out = Y_out_base.clone()

        # Initial recurrent states (zeros)
        S_phi = torch.zeros((B, H, m_O), device=Q.device, dtype=torch.float32)
        S_K   = torch.zeros((B, H, m_O, j_padded), device=Q.device, dtype=torch.float32)
        S_KV  = torch.zeros((B, H, d_head, m_O, j_padded), device=Q.device, dtype=torch.float32)

        grid = (B, H, d_head)
        r = outlier_idx.shape[1]
        j = inlier_idx.shape[1]

        for u in range(U):
            hybrid_single_chunk_kernel[grid](
                Q, K, V, Y_out,
                omega_O,
                S_phi, S_K, S_KV,
                outlier_idx, inlier_idx,
                u, N, r, j, inv_sqrt_m_O,      # <-- FIXED: only 5 scalars, no extra d_head
                C=C, r_padded=r_padded, j_padded=j_padded, m_O=m_O,
                inlier_scale=inlier_scale,
                stride_q_b=Q.stride(0), stride_q_h=Q.stride(1), stride_q_n=Q.stride(2), stride_q_d=Q.stride(3),
                stride_k_b=K.stride(0), stride_k_h=K.stride(1), stride_k_n=K.stride(2), stride_k_d=K.stride(3),
                stride_v_b=V.stride(0), stride_v_h=V.stride(1), stride_v_n=V.stride(2), stride_v_d=V.stride(3),
                stride_yout_b=Y_out.stride(0), stride_yout_h=Y_out.stride(1), stride_yout_n=Y_out.stride(2), stride_yout_d=Y_out.stride(3),
                stride_om_r=omega_O.stride(0), stride_om_m=omega_O.stride(1),
                stride_sphi_b=S_phi.stride(0), stride_sphi_h=S_phi.stride(1), stride_sphi_m=S_phi.stride(2),
                stride_sk_b=S_K.stride(0), stride_sk_h=S_K.stride(1), stride_sk_m=S_K.stride(2), stride_sk_j=S_K.stride(3),
                stride_skv_b=S_KV.stride(0), stride_skv_h=S_KV.stride(1), stride_skv_d=S_KV.stride(2), stride_skv_m=S_KV.stride(3), stride_skv_j=S_KV.stride(4),
                stride_oi_h=outlier_idx.stride(0), stride_oi_r=outlier_idx.stride(1),
                stride_ii_h=inlier_idx.stride(0), stride_ii_j=inlier_idx.stride(1),
                num_warps=4, num_stages=2,
            )

        # Save for backward
        ctx.save_for_backward(
            Q, K, V, Y_out_base,
            omega_O, outlier_idx, inlier_idx,
            S_phi, S_K, S_KV  # final states
        )
        ctx.C = C
        ctx.r_padded = r_padded
        ctx.j_padded = j_padded
        ctx.m_O = m_O
        ctx.inlier_scale = inlier_scale
        ctx.inv_sqrt_m_O = inv_sqrt_m_O
        ctx.N = N
        ctx.pad_len = pad_len
        ctx.U = U
        ctx.d_head = d_head

        return Y_out

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, Y_out_base, omega_O, outlier_idx, inlier_idx, S_phi_final, S_K_final, S_KV_final = ctx.saved_tensors
        orig_dtype = Q.dtype
        Q = Q.float()
        K = K.float()
        V = V.float()
        Y_out_base = Y_out_base.float()
        omega_O = omega_O.float()
        S_phi_final = S_phi_final.float()
        S_K_final = S_K_final.float()
        S_KV_final = S_KV_final.float()
        grad_output = grad_output.float()
        C = ctx.C
        r_padded = ctx.r_padded
        j_padded = ctx.j_padded
        m_O = ctx.m_O
        inlier_scale = ctx.inlier_scale
        inv_sqrt_m_O = ctx.inv_sqrt_m_O
        N = ctx.N
        pad_len = ctx.pad_len
        U = ctx.U
        d_head = ctx.d_head

        B, H, N_pad, _ = Q.shape
        device = Q.device

        # Initialize gradients for inputs
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)
        grad_Y_out_base = torch.zeros_like(Y_out_base)
        grad_omega_O = torch.zeros(r_padded, m_O, device=Q.device, dtype=Q.dtype)

        # Current states (starting from final states)
        S_phi = S_phi_final
        S_K   = S_K_final
        S_KV  = S_KV_final

        # Gradients w.r.t. state after the current chunk (downstream gradient)
        grad_S_phi = torch.zeros_like(S_phi)
        grad_S_K   = torch.zeros_like(S_K)
        grad_S_KV  = torch.zeros_like(S_KV)

        # Pad omega_O to r_padded
        omega_O_pad = F.pad(omega_O, (0, 0, 0, r_padded - omega_O.shape[0]))
        omega_O_pad = omega_O_pad.to(Q.dtype)

        # Expand indices for gathering
        r = outlier_idx.shape[1]
        j = inlier_idx.shape[1]

        outlier_idx_exp = outlier_idx[None, :, None, :].expand(B, H, C, r)
        inlier_idx_exp  = inlier_idx[None, :, None, :].expand(B, H, C, j)

        # Mask for valid inlier columns (FIX: zero out padded dimensions)
        mask_j = (torch.arange(j_padded, device=device) < j).view(1, 1, 1, -1)

        for u in reversed(range(U)):
            start = u * C
            end = start + C

            # Valid token mask for this chunk
            mask_valid = (torch.arange(C, device=device) + start < N).float()
            mask_valid = mask_valid.view(1, 1, C, 1)

            # Extract slices
            Q_chunk = Q[:, :, start:end, :]
            K_chunk = K[:, :, start:end, :]
            V_chunk = V[:, :, start:end, :]
            Y_base_chunk = Y_out_base[:, :, start:end, :]

            # Gather outlier and inlier features
            Q_O = Q_chunk.gather(dim=-1, index=outlier_idx_exp)
            K_O = K_chunk.gather(dim=-1, index=outlier_idx_exp)
            Q_O = torch.nn.functional.pad(Q_O, (0, r_padded - r))
            K_O = torch.nn.functional.pad(K_O, (0, r_padded - r))
            Q_J_gathered = Q_chunk.gather(dim=-1, index=inlier_idx_exp)  # (B,H,C,j)
            K_J_gathered = K_chunk.gather(dim=-1, index=inlier_idx_exp)

            # Pad and then mask the inlier features (FIX)
            Q_J = F.pad(Q_J_gathered, (0, j_padded - j)) * mask_j
            K_J = F.pad(K_J_gathered, (0, j_padded - j)) * mask_j

            V_d = V_chunk
            Y_O_d = Y_base_chunk

            # ----- Recompute phi_Q, phi_K -----
            Q_O_omega = torch.matmul(Q_O, omega_O_pad)
            K_O_omega = torch.matmul(K_O, omega_O_pad)
            Q_O_norm = 0.5 * (Q_O ** 2).sum(dim=-1, keepdim=True)
            K_O_norm = 0.5 * (K_O ** 2).sum(dim=-1, keepdim=True)
            phi_Q = torch.exp(Q_O_omega - Q_O_norm) * inv_sqrt_m_O
            phi_K = torch.exp(K_O_omega - K_O_norm) * inv_sqrt_m_O

            # ----- State update deltas (masked for valid tokens) -----
            mask_v = mask_valid.squeeze(-1).unsqueeze(-1)
            phi_K_masked = phi_K * mask_v
            K_J_masked = K_J * mask_v
            V_d_masked = V_d * mask_v

            K_J_scaled = K_J_masked / inlier_scale

            delta_phi = phi_K_masked.sum(dim=2)  # (B,H,m_O)
            delta_K   = torch.einsum('bhcm,bhcj->bhmj', phi_K_masked, K_J_scaled)
            delta_KV  = torch.einsum('bhcm,bhcj,bhcd->bhdmj', phi_K_masked, K_J_scaled, V_d_masked)

            # Recover prior states
            S_phi_prev = S_phi - delta_phi
            S_K_prev   = S_K - delta_K
            S_KV_prev  = S_KV - delta_KV

            # ----- Forward pass using prior states -----
            logits_local = torch.matmul(Q_O, K_O.transpose(-2, -1))
            causal_mask = torch.tril(torch.ones(C, C, device=device, dtype=torch.bool)).view(1, 1, C, C)
            logits_local = logits_local.masked_fill(~causal_mask, float('-inf'))
            P_local = torch.softmax(logits_local, dim=-1)

            weighted_K_J = torch.matmul(P_local, K_J)
            e_local = (Q_J * weighted_K_J).sum(dim=-1) / inlier_scale

            E = torch.matmul(Q_J, K_J.transpose(-2, -1)) / inlier_scale
            P_local_E = P_local * E
            term1_local = torch.einsum('bhij,bhjd->bhid', P_local_E, V_d)

            Z_past = torch.einsum('bhcm,bhm->bhc', phi_Q, S_phi_prev)
            attn_phi = torch.matmul(phi_Q, phi_K.transpose(-2, -1))
            attn_phi_causal = attn_phi.masked_fill(~causal_mask, 0.0)
            Z_local = attn_phi_causal.sum(dim=-1)
            Z_total = Z_past + Z_local + 1e-8

            W_K = torch.einsum('bhcm,bhmj->bhcj', phi_Q, S_K_prev)
            e_past_num = (Q_J * W_K).sum(dim=-1)
            e_past = e_past_num / Z_total
            e = e_local + e_past
            e_clamped = torch.clamp(e, -0.95, 0.95)

            W_KV = torch.einsum('bhcm,bhdmj->bhcdj', phi_Q, S_KV_prev)
            t1p_num = (Q_J.unsqueeze(-2) * W_KV).sum(dim=-1)
            term1_past = t1p_num / Z_total.unsqueeze(-1)

            # ----- Gradient of this chunk -----
            grad_y_chunk = grad_output[:, :, start:end, :] * mask_valid

            grad_Y_O_d = grad_y_chunk * (1.0 - e_clamped.unsqueeze(-1))
            grad_term1_local = grad_y_chunk
            grad_term1_past = grad_y_chunk
            grad_e_clamped = -(grad_y_chunk * Y_O_d).sum(dim=-1)

            # Clamp backward
            grad_e = grad_e_clamped * ((e > -0.95) & (e < 0.95)).float()
            grad_e_local = grad_e
            grad_e_past = grad_e

            # e_local backward
            grad_e_local_scaled = grad_e_local / inlier_scale
            grad_Q_J_e_local = grad_e_local_scaled.unsqueeze(-1) * weighted_K_J
            grad_weighted_K_J = grad_e_local_scaled.unsqueeze(-1) * Q_J

            # weighted_K_J backward
            grad_P_local_wKJ = torch.matmul(grad_weighted_K_J, K_J.transpose(-2, -1))
            grad_K_J_wKJ = torch.matmul(P_local.transpose(-2, -1), grad_weighted_K_J)

            # term1_past backward
            grad_t1p_num = grad_term1_past / Z_total.unsqueeze(-1)
            grad_Z_total_past = -(grad_t1p_num * term1_past).sum(dim=-1)
            grad_Q_J_past = torch.einsum('bhcd,bhcdj->bhcj', grad_t1p_num, W_KV)
            grad_W_KV = grad_t1p_num.unsqueeze(-1) * Q_J.unsqueeze(-2)

            # e_past backward
            grad_e_past_num = grad_e_past / Z_total
            grad_Z_total_e = -grad_e_past_num * e_past
            grad_Q_J_e_past = grad_e_past_num.unsqueeze(-1) * W_K
            grad_W_K = grad_e_past_num.unsqueeze(-1) * Q_J

            # Total grad_Z_total
            grad_Z_total = grad_Z_total_past + grad_Z_total_e
            grad_Z_past = grad_Z_total
            grad_Z_local = grad_Z_total

            # Z_local backward
            grad_attn_phi_causal = grad_Z_local.unsqueeze(-1).expand_as(attn_phi_causal) * causal_mask.float()
            grad_attn_phi = grad_attn_phi_causal
            grad_phi_Q_attn = torch.matmul(grad_attn_phi, phi_K)
            grad_phi_K_attn = torch.matmul(grad_attn_phi.transpose(-2, -1), phi_Q)

            # Z_past backward
            grad_phi_Q_Zpast = grad_Z_past.unsqueeze(-1) * S_phi_prev.unsqueeze(2)
            grad_S_phi_prev = (grad_Z_past.unsqueeze(-1) * phi_Q).sum(dim=2)

            # W_K backward
            grad_phi_Q_WK = torch.einsum('bhcj,bhmj->bhcm', grad_W_K, S_K_prev)
            grad_S_K_prev = torch.einsum('bhcm,bhcj->bhmj', phi_Q, grad_W_K)

            # W_KV backward
            grad_phi_Q_WKV = torch.einsum('bhcdj,bhdmj->bhcm', grad_W_KV, S_KV_prev)
            grad_S_KV_prev = torch.einsum('bhcm,bhcdj->bhdmj', phi_Q, grad_W_KV)

            # Accumulate total phi_Q and phi_K gradients (local part)
            grad_phi_Q = grad_phi_Q_attn + grad_phi_Q_Zpast + grad_phi_Q_WK + grad_phi_Q_WKV
            grad_phi_K = grad_phi_K_attn

            # ----- State update gradients (from downstream) -----
            grad_delta_phi = grad_S_phi
            grad_delta_K   = grad_S_K
            grad_delta_KV  = grad_S_KV

            grad_phi_K_masked = grad_delta_phi.unsqueeze(2).expand_as(phi_K_masked)
            grad_phi_K_masked = grad_phi_K_masked + torch.einsum('bhmj,bhcj->bhcm', grad_delta_K, K_J_scaled)
            grad_K_J_scaled_delta_K = torch.einsum('bhcm,bhmj->bhcj', phi_K_masked, grad_delta_K)

            grad_phi_K_masked += torch.einsum('bhdmj,bhcj,bhcd->bhcm', grad_delta_KV, K_J_scaled, V_d_masked)
            grad_K_J_scaled_delta_KV = torch.einsum('bhdmj,bhcm,bhcd->bhcj', grad_delta_KV, phi_K_masked, V_d_masked)
            grad_V_d_delta_KV = torch.einsum('bhdmj,bhcm,bhcj->bhcd', grad_delta_KV, phi_K_masked, K_J_scaled)

            grad_phi_K += grad_phi_K_masked  # mask_v already applied, gradient flows back

            grad_K_J_scaled = grad_K_J_scaled_delta_K + grad_K_J_scaled_delta_KV
            grad_K_J_from_scaled = grad_K_J_scaled / inlier_scale

            grad_V_d = grad_V_d_delta_KV

            # ----- P_local_E backward -----
            grad_P_local_E = torch.einsum('bhid,bhjd->bhij', grad_term1_local, V_d)
            grad_V_d_term1 = torch.einsum('bhij,bhid->bhjd', P_local_E, grad_term1_local)
            grad_V_d += grad_V_d_term1

            grad_P_local_E_via = grad_P_local_E * E
            grad_E = grad_P_local_E * P_local

            # Total grad_P_local
            grad_P_local = grad_P_local_wKJ + grad_P_local_E_via

            # Softmax backward
            S_soft = (grad_P_local * P_local).sum(dim=-1, keepdim=True)
            grad_logits_local = P_local * (grad_P_local - S_soft)

            grad_Q_O_logits = torch.matmul(grad_logits_local, K_O)
            grad_K_O_logits = torch.matmul(grad_logits_local.transpose(-2, -1), Q_O)

            # E backward -> Q_J, K_J
            grad_Q_J_E = torch.matmul(grad_E, K_J) / inlier_scale
            grad_K_J_E = torch.matmul(grad_E.transpose(-2, -1), Q_J) / inlier_scale

            # Total Q_J and K_J gradients
            grad_Q_J = grad_Q_J_e_local + grad_Q_J_past + grad_Q_J_e_past + grad_Q_J_E
            grad_K_J = grad_K_J_wKJ + grad_K_J_E + grad_K_J_from_scaled

            # ----- phi_Q, phi_K backward to Q_O, K_O, omega_O -----
            grad_S_Q = grad_phi_Q * phi_Q
            grad_norm_Q = -grad_S_Q.sum(dim=-1)
            grad_Q_O_norm = grad_norm_Q.unsqueeze(-1) * Q_O
            grad_Q_O_omega = torch.matmul(grad_S_Q, omega_O_pad.transpose(-2, -1))
            grad_omega_O += torch.einsum('bhcr,bhcm->rm', Q_O, grad_S_Q)

            grad_S_K_temp = grad_phi_K * phi_K
            grad_norm_K = -grad_S_K_temp.sum(dim=-1)
            grad_K_O_norm = grad_norm_K.unsqueeze(-1) * K_O
            grad_K_O_omega = torch.matmul(grad_S_K_temp, omega_O_pad.transpose(-2, -1))
            grad_omega_O += torch.einsum('bhcr,bhcm->rm', K_O, grad_S_K_temp)

            grad_Q_O = grad_Q_O_logits + grad_Q_O_norm + grad_Q_O_omega
            grad_K_O = grad_K_O_logits + grad_K_O_norm + grad_K_O_omega

            # Restrict inlier gradients to actual feature size j
            grad_Q_J_unpadded = grad_Q_J[:, :, :, :j]
            grad_K_J_unpadded = grad_K_J[:, :, :, :j]

            # ----- Scatter gradients to full tensors -----
            grad_Q[:, :, start:end, :].scatter_add_(-1, outlier_idx_exp, grad_Q_O[:, :, :, :r])

            grad_Q[:, :, start:end, :].scatter_add_(-1, inlier_idx_exp, grad_Q_J_unpadded)

            grad_K[:, :, start:end, :].scatter_add_(-1, outlier_idx_exp, grad_K_O[:, :, :, :r])
            grad_K[:, :, start:end, :].scatter_add_(-1, inlier_idx_exp, grad_K_J_unpadded)

            grad_V[:, :, start:end, :] += grad_V_d

            grad_Y_out_base[:, :, start:end, :] += grad_Y_O_d

            # ----- Prepare states for previous chunk -----
            grad_S_phi_prev_total = grad_S_phi_prev + grad_S_phi
            grad_S_K_prev_total   = grad_S_K_prev + grad_S_K
            grad_S_KV_prev_total  = grad_S_KV_prev + grad_S_KV

            S_phi, S_K, S_KV = S_phi_prev, S_K_prev, S_KV_prev
            grad_S_phi = grad_S_phi_prev_total
            grad_S_K   = grad_S_K_prev_total
            grad_S_KV  = grad_S_KV_prev_total

        grad_omega_O = grad_omega_O[:omega_O.shape[0], :]

        return (
            grad_Q.to(orig_dtype), grad_K.to(orig_dtype), grad_V.to(orig_dtype), grad_Y_out_base.to(orig_dtype), grad_omega_O.to(orig_dtype),
            None, None, None, None, None, None, None, None, None, None
        )


# -----------------------------------------------------------------------------
# Updated HybridOutlierFactorizedAttention module
# -----------------------------------------------------------------------------
class HybridOutlierFactorizedAttention(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.d_model   = model_cfg.d_model
        self.num_heads = model_cfg.num_heads
        self.d_head    = model_cfg.d_head
        self.r         = model_cfg.r
        self.m         = model_cfg.m
        self.m_O       = model_cfg.m_O
        self.j         = self.d_head - self.r
        self.chunk_size = model_cfg.chunk_size

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
        inlier_scale = float(self.j ** 0.5)

        Q = (self.W_q(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2).contiguous()
        K = (self.W_k(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2).contiguous()
        V = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2).contiguous()

        if self.r == 0:
            phi_Q = torch.exp(Q @ self.omega_J.to(dtype_in) -
                              (Q ** 2).sum(dim=-1, keepdim=True) / 2.0) / math.sqrt(self.m)
            phi_K = torch.exp(K @ self.omega_J.to(dtype_in) -
                              (K ** 2).sum(dim=-1, keepdim=True) / 2.0) / math.sqrt(self.m)
            KV = torch.einsum('bhnm, bhnd -> bhnmd', phi_K, V)
            KV_cum = torch.cumsum(KV, dim=2)
            K_cum  = torch.cumsum(phi_K, dim=2)
            num = torch.einsum('bhnm, bhnmd -> bhnd', phi_Q, KV_cum)
            den = torch.einsum('bhnm, bhnm -> bhn', phi_Q, K_cum).unsqueeze(-1)
            Y = num / (den + 1e-8)
            Y = Y.transpose(1, 2).reshape(B, N, D)
            return self.out_proj(Y).to(dtype_in)

        if self.r == self.d_head:
            Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=1.0)
            Y = Y.transpose(1, 2).reshape(B, N, D)
            return self.out_proj(Y).to(dtype_in)

        self._maybe_update_indices()
        outlier_idx = self._cached_outlier_idx
        inlier_idx  = self._cached_inlier_idx

        out_gather = outlier_idx.view(1, self.num_heads, 1, self.r).expand(B, self.num_heads, N, self.r)
        Q_O_tmp = Q.gather(-1, out_gather)
        K_O_tmp = K.gather(-1, out_gather)
        Y_out = F.scaled_dot_product_attention(Q_O_tmp, K_O_tmp, V, is_causal=True, scale=1.0)
        del Q_O_tmp, K_O_tmp

        C = self.chunk_size
        pad_len = (C - (N % C)) % C
        if pad_len > 0:
            Q = F.pad(Q, (0, 0, 0, pad_len))
            K = F.pad(K, (0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, pad_len))
            Y_out = F.pad(Y_out, (0, 0, 0, pad_len))

        r_padded = max(16, int(2 ** math.ceil(math.log2(self.r)))) if self.r > 0 else 16
        j_padded = max(16, int(2 ** math.ceil(math.log2(self.j)))) if self.j > 0 else 16
        m_O = self.m_O
        inv_sqrt_m_O = 1.0 / math.sqrt(m_O)

        Y_out = HOFAFunction.apply(
            Q, K, V, Y_out,
            self.omega_O,
            outlier_idx, inlier_idx,
            C, r_padded, j_padded, m_O,
            inlier_scale, inv_sqrt_m_O,
            N, pad_len
        )

        if pad_len > 0:
            Y_out = Y_out[:, :, :-pad_len]

        Y_out = Y_out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(Y_out).to(dtype_in)


# -----------------------------------------------------------------------------
# Transformer block and language model (unchanged)
# -----------------------------------------------------------------------------
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