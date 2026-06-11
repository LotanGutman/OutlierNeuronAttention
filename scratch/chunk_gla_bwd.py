import torch
import triton
import triton.language as tl

@triton.jit
def chunk_gla_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, gamma_ptr, inlier_idx_ptr,
    states_in_ptr, dY_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    N, chunk_size: tl.constexpr, 
    j, j_padded: tl.constexpr, d_head, d_head_padded: tl.constexpr,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_gamma_b, stride_gamma_h, stride_gamma_n,
    stride_inlier_h,
    stride_state_u, stride_state_in_b, stride_state_in_h, stride_state_in_j, stride_state_in_d,
    stride_dy_b, stride_dy_h, stride_dy_n, stride_dy_d
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offsets_j = tl.arange(0, j_padded)
    offsets_d = tl.arange(0, d_head_padded)
    offsets_c = tl.arange(0, chunk_size)

    mask_j = offsets_j < j
    mask_d = offsets_d < d_head

    inlier_idx_offset = pid_h * stride_inlier_h
    inlier_indices = tl.load(inlier_idx_ptr + inlier_idx_offset + offsets_j, mask=mask_j, other=0)

    b_h_q_ptr = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    b_h_k_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    b_h_v_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h
    b_h_gamma_ptr = gamma_ptr + pid_b * stride_gamma_b + pid_h * stride_gamma_h
    
    b_h_dy_ptr = dY_ptr + pid_b * stride_dy_b + pid_h * stride_dy_h
    
    b_h_dq_ptr = dQ_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    b_h_dk_ptr = dK_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    b_h_dv_ptr = dV_ptr + pid_b * stride_v_b + pid_h * stride_v_h

    b_h_state_in_ptr = states_in_ptr + pid_b * stride_state_in_b + pid_h * stride_state_in_h

    U = tl.cdiv(N, chunk_size)
    dS = tl.zeros([j_padded, d_head_padded], dtype=tl.float32)

    for u in range(U - 1, -1, -1):
        start = u * chunk_size
        t_offs = start + offsets_c
        mask_t = t_offs < N

        # Load S_u (state entering this chunk)
        state_in_ptrs = b_h_state_in_ptr + u * stride_state_u + offsets_j[:, None] * stride_state_in_j + offsets_d[None, :] * stride_state_in_d
        S_u = tl.load(state_in_ptrs, mask=mask_j[:, None] & mask_d[None, :], other=0.0)

        q_ptrs = b_h_q_ptr + t_offs[:, None] * stride_q_n + inlier_indices[None, :] * stride_q_d
        Q_c = tl.load(q_ptrs, mask=mask_t[:, None] & mask_j[None, :], other=0.0)

        k_ptrs = b_h_k_ptr + t_offs[:, None] * stride_k_n + inlier_indices[None, :] * stride_k_d
        K_c = tl.load(k_ptrs, mask=mask_t[:, None] & mask_j[None, :], other=0.0)

        v_ptrs = b_h_v_ptr + t_offs[:, None] * stride_v_n + offsets_d[None, :] * stride_v_d
        V_c = tl.load(v_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0)

        gamma_ptrs = b_h_gamma_ptr + t_offs * stride_gamma_n
        gamma_c = tl.load(gamma_ptrs, mask=mask_t, other=1.0).to(tl.float32)
        
        dy_ptrs = b_h_dy_ptr + t_offs[:, None] * stride_dy_n + offsets_d[None, :] * stride_dy_d
        dY_c = tl.load(dy_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0)

        # Forward recompute
        g_c = tl.log(gamma_c)
        g_cumsum = tl.cumsum(g_c, axis=0)
        g_ex = g_cumsum - g_c
        diff = g_ex[:, None] - g_cumsum[None, :]
        mask = tl.exp(diff) * tl.where(offsets_c[:, None] >= offsets_c[None, :], 1.0, 0.0)
        
        last_idx = tl.max(tl.where(mask_t, offsets_c, 0))
        g_cumsum_last = tl.max(tl.where(offsets_c == last_idx, g_cumsum, -float('inf')))
        k_decay = tl.exp(g_cumsum_last - g_cumsum)
        K_c_decayed = K_c * k_decay[:, None] * tl.where(mask_t[:, None], 1.0, 0.0)
        
        attn = tl.dot(Q_c, tl.trans(K_c), allow_tf32=True)
        attn = (attn * mask).to(Q_c.dtype)

        # 1. dY_inter
        # Y_inter = Q_c S_u e^{g_ex}
        # dY_inter = dY_c
        dY_inter = dY_c * tl.exp(g_ex)[:, None]
        dQ_c = tl.dot(dY_inter, tl.trans(S_u).to(dY_inter.dtype), allow_tf32=True)
        dS_u_from_Y = tl.dot(tl.trans(Q_c), dY_inter, allow_tf32=True)

        # 2. dS from next chunk
        # S_{u+1} = S_u e^{g_cumsum_last} + K_c_decayed^T V_c
        dS_u = dS_u_from_Y + dS * tl.exp(g_cumsum_last)
        dK_c_decayed = tl.dot(V_c, tl.trans(dS).to(V_c.dtype), allow_tf32=True)
        dV_c = tl.dot(K_c_decayed, dS.to(K_c.dtype), allow_tf32=True)
        
        dK_c = dK_c_decayed * k_decay[:, None] * tl.where(mask_t[:, None], 1.0, 0.0)

        # 3. Y_intra
        # Y_intra = attn V_c
        d_attn = tl.dot(dY_c, tl.trans(V_c), allow_tf32=True)
        dV_c += tl.dot(tl.trans(attn), dY_c, allow_tf32=True)
        
        d_attn = d_attn * mask
        dQ_c += tl.dot(d_attn.to(K_c.dtype), K_c, allow_tf32=True)
        dK_c += tl.dot(tl.trans(d_attn.to(Q_c.dtype)), Q_c, allow_tf32=True)
        
        # Advance dS
        dS = dS_u

        # Store gradients
        dq_ptrs = b_h_dq_ptr + t_offs[:, None] * stride_q_n + inlier_indices[None, :] * stride_q_d
        tl.store(dq_ptrs, dQ_c.to(dQ_ptr.dtype.element_ty), mask=mask_t[:, None] & mask_j[None, :])
        
        dk_ptrs = b_h_dk_ptr + t_offs[:, None] * stride_k_n + inlier_indices[None, :] * stride_k_d
        tl.store(dk_ptrs, dK_c.to(dK_ptr.dtype.element_ty), mask=mask_t[:, None] & mask_j[None, :])
        
        dv_ptrs = b_h_dv_ptr + t_offs[:, None] * stride_v_n + offsets_d[None, :] * stride_v_d
        tl.store(dv_ptrs, dV_c.to(dV_ptr.dtype.element_ty), mask=mask_t[:, None] & mask_d[None, :])
