import torch
import triton
import triton.language as tl

@triton.jit
def chunk_gla_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, log_gamma_ptr,
    Log_Alpha_ptr, Mix_G_ptr, Gla_Scale_ptr,
    states_in_ptr,
    Y_ptr,
    states_out_ptr,
    N, chunk_size: tl.constexpr,
    r: tl.constexpr, j, j_padded: tl.constexpr, d_head, d_head_padded: tl.constexpr,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_log_gamma_b, stride_log_gamma_h, stride_log_gamma_n,
    stride_la_b, stride_la_h, stride_la_j,
    stride_mg_b, stride_mg_h, stride_mg_n,
    stride_gs_h, stride_gs_d,
    stride_state_u, stride_state_in_b, stride_state_in_h, stride_state_in_j, stride_state_in_d,
    stride_y_b, stride_y_h, stride_y_n, stride_y_d,
    stride_state_out_b, stride_state_out_h, stride_state_out_j, stride_state_out_d,
    HAS_ALPHA: tl.constexpr,
    FUSE_NORM_GATE: tl.constexpr,
    REQUIRES_GRAD: tl.constexpr,
    eps: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offsets_j = tl.arange(0, j_padded)
    offsets_d = tl.arange(0, d_head_padded)
    offsets_c = tl.arange(0, chunk_size)

    mask_j = offsets_j < j
    mask_d = offsets_d < d_head

    inlier_indices = r + offsets_j

    b_h_q_ptr = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    b_h_k_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    b_h_v_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h
    b_h_log_gamma_ptr = log_gamma_ptr + pid_b * stride_log_gamma_b + pid_h * stride_log_gamma_h
    b_h_y_ptr = Y_ptr + pid_b * stride_y_b + pid_h * stride_y_h

    if HAS_ALPHA:
        la_ptrs = Log_Alpha_ptr + pid_b * stride_la_b + pid_h * stride_la_h + offsets_j * stride_la_j
        alpha_j = tl.load(la_ptrs, mask=mask_j, other=0.0).to(tl.float32)
        alpha_vec = tl.exp(alpha_j)
    else:
        alpha_vec = 1.0

    if FUSE_NORM_GATE:
        b_h_mg_ptr = Mix_G_ptr + pid_b * stride_mg_b + pid_h * stride_mg_h
        gs_ptrs = Gla_Scale_ptr + pid_h * stride_gs_h + offsets_d * stride_gs_d
        gla_scale_vec = tl.load(gs_ptrs, mask=mask_d, other=0.0).to(tl.float32)

    if REQUIRES_GRAD:
        b_h_state_in_ptr = states_in_ptr + pid_b * stride_state_in_b + pid_h * stride_state_in_h

    U = tl.cdiv(N, chunk_size)
    S = tl.zeros([j_padded, d_head_padded], dtype=tl.float32)

    for u in range(U):
        start = u * chunk_size
        t_offs = start + offsets_c
        mask_t = t_offs < N

        if REQUIRES_GRAD:
            state_in_ptrs = b_h_state_in_ptr + u * stride_state_u + offsets_j[:, None] * stride_state_in_j + offsets_d[None, :] * stride_state_in_d
            tl.store(state_in_ptrs, S, mask=mask_j[:, None] & mask_d[None, :])

        q_ptrs = b_h_q_ptr + t_offs[:, None] * stride_q_n + inlier_indices[None, :] * stride_q_d
        Q_c = (tl.load(q_ptrs, mask=mask_t[:, None] & mask_j[None, :], other=0.0)).to(tl.float32)

        k_ptrs = b_h_k_ptr + t_offs[:, None] * stride_k_n + inlier_indices[None, :] * stride_k_d
        K_c = (tl.load(k_ptrs, mask=mask_t[:, None] & mask_j[None, :], other=0.0)).to(tl.float32)

        v_ptrs = b_h_v_ptr + t_offs[:, None] * stride_v_n + offsets_d[None, :] * stride_v_d
        V_c = tl.load(v_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0)

        log_gamma_ptrs = b_h_log_gamma_ptr + t_offs * stride_log_gamma_n
        g_c = tl.load(log_gamma_ptrs, mask=mask_t, other=0.0).to(tl.float32)

        g_cumsum = tl.cumsum(g_c, axis=0)
        last_idx = tl.max(tl.where(mask_t, offsets_c, 0))
        g_cumsum_last = tl.max(tl.where(offsets_c == last_idx, g_cumsum, -float('inf')))

        if HAS_ALPHA:
            # Numerically stable factored decay
            g_tilde = (g_cumsum - g_cumsum_last)[:, None] * alpha_vec[None, :] # (C, J) <= 0
            g_hat = (g_cumsum_last - g_cumsum)[:, None] * alpha_vec[None, :]   # (C, J) >= 0

            Q_c_factored = (Q_c * tl.math.exp(g_tilde) * tl.where(mask_t[:, None] & mask_j[None, :], 1.0, 0.0)).to(Q_ptr.dtype.element_ty)
            K_c_factored = (K_c * tl.math.exp(g_hat) * tl.where(mask_t[:, None] & mask_j[None, :], 1.0, 0.0)).to(Q_ptr.dtype.element_ty)

            attn = tl.dot(Q_c_factored, tl.trans(K_c_factored), allow_tf32=False)
            attn = tl.where(offsets_c[:, None] >= offsets_c[None, :], attn, 0.0).to(Q_ptr.dtype.element_ty)
            Y_intra = tl.dot(attn, V_c, allow_tf32=False)

            Q_c_inter = (Q_c * tl.math.exp(g_cumsum[:, None] * alpha_vec[None, :]) * tl.where(mask_t[:, None] & mask_j[None, :], 1.0, 0.0)).to(Q_ptr.dtype.element_ty)
            Y_inter = tl.dot(Q_c_inter, S.to(Q_ptr.dtype.element_ty), allow_tf32=False)

            S_update = tl.dot(tl.trans(K_c_factored), V_c, allow_tf32=False)
            decay_state = tl.math.exp(g_cumsum_last * alpha_vec)[:, None]
            S = S * decay_state + S_update
        else:
            diff = g_cumsum[:, None] - g_cumsum[None, :]
            diff = tl.where(offsets_c[:, None] >= offsets_c[None, :], diff, -float('inf'))
            mask = tl.math.exp(diff)

            Q_c_typed = (Q_c * tl.where(mask_t[:, None] & mask_j[None, :], 1.0, 0.0)).to(Q_ptr.dtype.element_ty)
            K_c_typed = (K_c * tl.where(mask_t[:, None] & mask_j[None, :], 1.0, 0.0)).to(Q_ptr.dtype.element_ty)

            attn = tl.dot(Q_c_typed, tl.trans(K_c_typed), allow_tf32=False)
            attn = (attn * mask).to(Q_ptr.dtype.element_ty)
            Y_intra = tl.dot(attn, V_c, allow_tf32=False)

            Y_inter = tl.dot(Q_c_typed, S.to(Q_ptr.dtype.element_ty), allow_tf32=False) * tl.math.exp(g_cumsum)[:, None]

            k_decay = tl.math.exp(g_cumsum_last - g_cumsum)
            K_c_decayed = (K_c * k_decay[:, None] * tl.where(mask_t[:, None] & mask_j[None, :], 1.0, 0.0)).to(Q_ptr.dtype.element_ty)

            S_update = tl.dot(tl.trans(K_c_decayed), V_c, allow_tf32=False)
            S = S * tl.math.exp(g_cumsum_last) + S_update

        Y_total = (Y_intra + Y_inter).to(tl.float32)

        if FUSE_NORM_GATE:
            Y_masked = tl.where(mask_d[None, :], Y_total, 0.0)
            var = tl.sum(Y_masked * Y_masked, axis=1) / d_head
            rsqrt = tl.math.rsqrt(var + eps)
            Y_norm = Y_total * rsqrt[:, None] * gla_scale_vec[None, :]

            mg_ptrs = b_h_mg_ptr + t_offs * stride_mg_n
            mg_val = tl.load(mg_ptrs, mask=mask_t, other=0.0).to(tl.float32)
            blend = 1.0 - mg_val
            Y_store = Y_norm * blend[:, None]
        else:
            Y_store = Y_total

        y_ptrs = b_h_y_ptr + t_offs[:, None] * stride_y_n + offsets_d[None, :] * stride_y_d
        tl.store(y_ptrs, Y_store.to(Y_ptr.dtype.element_ty), mask=mask_t[:, None] & mask_d[None, :])

    # Export final recurrent state
    state_out_ptrs = states_out_ptr \
        + pid_b * stride_state_out_b + pid_h * stride_state_out_h \
        + offsets_j[:, None] * stride_state_out_j + offsets_d[None, :] * stride_state_out_d
    tl.store(state_out_ptrs, S, mask=mask_j[:, None] & mask_d[None, :])


def chunk_gla_inlier_fwd(Q, K, V, log_gamma, r, chunk_size,
                         log_alpha=None, mix_g=None, gla_scale=None, out=None,
                         states_in=None, requires_grad=False):
    B, H, N, d_head = Q.shape
    j = d_head - r
    device = Q.device

    j_padded = 1 << (j - 1).bit_length()
    j_padded = max(j_padded, 16)
    d_head_padded = 1 << (d_head - 1).bit_length()
    d_head_padded = max(d_head_padded, 16)

    Y = out if out is not None else torch.empty(B, H, N, d_head, dtype=Q.dtype, device=device)
    states_out = torch.empty(B, H, j_padded, d_head_padded, dtype=torch.float32, device=device)
    if states_in is None:
        states_in = torch.empty(1, 1, 1, 1, 1, device=device, dtype=torch.float32)

    HAS_ALPHA = log_alpha is not None
    if HAS_ALPHA:
        if log_alpha.ndim == 4:
            log_alpha_b = log_alpha.stride(0) if log_alpha.shape[0] > 1 else 0
            log_alpha_h = log_alpha.stride(1)
            log_alpha_j = log_alpha.stride(3)
        elif log_alpha.ndim == 2:
            log_alpha_b = 0
            log_alpha_h = log_alpha.stride(0)
            log_alpha_j = log_alpha.stride(1)
        else:
            log_alpha_c = log_alpha.contiguous()
            log_alpha_b, log_alpha_h, log_alpha_j = 0, log_alpha_c.shape[-1], 1
        log_alpha_tensor = log_alpha
    else:
        log_alpha_tensor = Q
        log_alpha_b, log_alpha_h, log_alpha_j = 0, 0, 0

    FUSE_NORM_GATE = (mix_g is not None) and (gla_scale is not None)
    if FUSE_NORM_GATE:
        mix_g_b = mix_g.stride(0)
        mix_g_h = mix_g.stride(1)
        mix_g_n = mix_g.stride(2)
        mix_g_tensor = mix_g

        if gla_scale.ndim == 4:
            gla_scale_h = gla_scale.stride(1)
            gla_scale_d = gla_scale.stride(3)
        elif gla_scale.ndim == 2:
            gla_scale_h = gla_scale.stride(0)
            gla_scale_d = gla_scale.stride(1)
        else:
            gla_scale_c = gla_scale.contiguous()
            gla_scale_h, gla_scale_d = gla_scale_c.shape[-1], 1
        gla_scale_tensor = gla_scale
    else:
        mix_g_tensor = Q
        gla_scale_tensor = Q
        mix_g_b, mix_g_h, mix_g_n = 0, 0, 0
        gla_scale_h, gla_scale_d = 0, 0

    grid = (B, H)

    chunk_gla_fwd_kernel[grid](
        Q, K, V, log_gamma,
        log_alpha_tensor, mix_g_tensor, gla_scale_tensor,
        states_in,
        Y,
        states_out,
        N, chunk_size,
        r, j, j_padded, d_head, d_head_padded,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        log_gamma.stride(0), log_gamma.stride(1), log_gamma.stride(2),
        log_alpha_b, log_alpha_h, log_alpha_j,
        mix_g_b, mix_g_h, mix_g_n,
        gla_scale_h, gla_scale_d,
        states_in.stride(2), states_in.stride(0), states_in.stride(1), states_in.stride(3), states_in.stride(4),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        states_out.stride(0), states_out.stride(1), states_out.stride(2), states_out.stride(3),
        HAS_ALPHA=HAS_ALPHA,
        FUSE_NORM_GATE=FUSE_NORM_GATE,
        REQUIRES_GRAD=requires_grad,
        eps=1e-5
    )

    return Y, states_out
