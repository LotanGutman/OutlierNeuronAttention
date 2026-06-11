import torch
import triton
import triton.language as tl

@triton.jit
def chunk_gla_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, gamma_ptr, inlier_idx_ptr,
    states_in_ptr,           
    Y_ptr,                   
    N, chunk_size: tl.constexpr, 
    j, j_padded: tl.constexpr, d_head, d_head_padded: tl.constexpr,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_gamma_b, stride_gamma_h, stride_gamma_n,
    stride_inlier_h,
    stride_state_u, stride_state_in_b, stride_state_in_h, stride_state_in_j, stride_state_in_d,
    stride_y_b, stride_y_h, stride_y_n, stride_y_d,
    scale,
    REQUIRES_GRAD: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offsets_j = tl.arange(0, j_padded)
    offsets_d = tl.arange(0, d_head_padded)
    offsets_c = tl.arange(0, chunk_size)

    mask_j = offsets_j < j
    mask_d = offsets_d < d_head

    # load actual dimension indices for this head
    inlier_idx_offset = pid_h * stride_inlier_h
    inlier_indices = tl.load(inlier_idx_ptr + inlier_idx_offset + offsets_j, mask=mask_j, other=0)

    b_h_q_ptr = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    b_h_k_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    b_h_v_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h
    b_h_gamma_ptr = gamma_ptr + pid_b * stride_gamma_b + pid_h * stride_gamma_h
    b_h_y_ptr = Y_ptr + pid_b * stride_y_b + pid_h * stride_y_h

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
        Q_c = (tl.load(q_ptrs, mask=mask_t[:, None] & mask_j[None, :], other=0.0) * scale).to(Q_ptr.dtype.element_ty)

        k_ptrs = b_h_k_ptr + t_offs[:, None] * stride_k_n + inlier_indices[None, :] * stride_k_d
        K_c = tl.load(k_ptrs, mask=mask_t[:, None] & mask_j[None, :], other=0.0)

        v_ptrs = b_h_v_ptr + t_offs[:, None] * stride_v_n + offsets_d[None, :] * stride_v_d
        V_c = tl.load(v_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0)

        gamma_ptrs = b_h_gamma_ptr + t_offs * stride_gamma_n
        gamma_c = tl.load(gamma_ptrs, mask=mask_t, other=1.0).to(tl.float32)

        g_c = tl.math.log2(gamma_c)
        g_cumsum = tl.cumsum(g_c, axis=0)

        diff = g_cumsum[:, None] - g_cumsum[None, :]
        mask = tl.math.exp2(diff) * tl.where(offsets_c[:, None] >= offsets_c[None, :], 1.0, 0.0)

        attn = tl.dot(Q_c, tl.trans(K_c), allow_tf32=False)
        attn = (attn * mask).to(Q_c.dtype)
        Y_intra = tl.dot(attn, V_c, allow_tf32=False)

        Y_inter = tl.dot(Q_c, S.to(Q_c.dtype), allow_tf32=False) * tl.math.exp2(g_cumsum)[:, None]
        Y_total = Y_intra + Y_inter

        y_ptrs = b_h_y_ptr + t_offs[:, None] * stride_y_n + offsets_d[None, :] * stride_y_d
        tl.store(y_ptrs, Y_total.to(Y_ptr.dtype.element_ty), mask=mask_t[:, None] & mask_d[None, :])

        last_idx = tl.max(tl.where(mask_t, offsets_c, 0))
        g_cumsum_last = tl.max(tl.where(offsets_c == last_idx, g_cumsum, -float('inf')))

        k_decay = tl.math.exp2(g_cumsum_last - g_cumsum)
        K_c_decayed = K_c * k_decay[:, None] * tl.where(mask_t[:, None], 1.0, 0.0)
        
        S_update = tl.dot(tl.trans(K_c_decayed.to(Q_c.dtype)), V_c, allow_tf32=False)
        S = S * tl.math.exp2(g_cumsum_last) + S_update

class ChunkGLAInlier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, gamma, inlier_idx, chunk_size=64):
        # Q, K are full (B, H, N, D), V is (B, H, N, d_head)
        B, H, N, _ = Q.shape
        d_head = V.shape[-1]
        j = inlier_idx.shape[-1]
        device = Q.device
        
        j_padded = 1 << (j - 1).bit_length()
        j_padded = max(j_padded, 16)
        d_head_padded = 1 << (d_head - 1).bit_length()
        d_head_padded = max(d_head_padded, 16)
        
        Y = torch.zeros(B, H, N, d_head, dtype=Q.dtype, device=device)
        U = (N + chunk_size - 1) // chunk_size
        
        requires_grad = False
        if requires_grad:
            states_in = torch.empty(U, B, H, j, d_head, dtype=torch.float32, device=device)
            stride_state_u = states_in.stride(0)
            stride_state_b = states_in.stride(1)
            stride_state_h = states_in.stride(2)
            stride_state_j = states_in.stride(3)
            stride_state_d = states_in.stride(4)
        else:
            states_in = torch.empty(1, device=device)
            stride_state_u = 0
            stride_state_b = 0
            stride_state_h = 0
            stride_state_j = 0
            stride_state_d = 0
        
        # Q, K, V are used as-is, strides are handled natively by Triton
        grid = (B, H)
        scale = d_head ** -0.5
        
        chunk_gla_fwd_kernel[grid](
            Q, K, V, gamma, inlier_idx,
            states_in, Y,
            N, chunk_size, j, j_padded, d_head, d_head_padded,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            gamma.stride(0), gamma.stride(1), gamma.stride(2),
            inlier_idx.stride(0), # stride_inlier_h
            stride_state_u, stride_state_b, stride_state_h, stride_state_j, stride_state_d,
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            scale,
            REQUIRES_GRAD=requires_grad
        )
            
        # ctx.save_for_backward(Q, K, V, gamma, states_in, inlier_idx)
        ctx.chunk_size = chunk_size
        ctx.j_padded = j_padded
        ctx.d_head_padded = d_head_padded
        return Y

    @staticmethod
    def backward(ctx, dY):
        return None, None, None, None, None, None
        
        return dQ, dK, dV, dgamma, None, None
