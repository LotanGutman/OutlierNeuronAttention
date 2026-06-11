import torch
import triton
import triton.language as tl

@triton.jit
def chunk_gla_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, gamma_ptr,
    states_in_ptr,           # (U, B, H, j, d_head)
    Y_ptr,                   # (B, H, N, d_head)
    N, chunk_size: tl.constexpr, 
    j, j_padded: tl.constexpr, d_head, d_head_padded: tl.constexpr,
    stride_q_b, stride_q_h, stride_q_n, stride_q_j,
    stride_k_b, stride_k_h, stride_k_n, stride_k_j,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_gamma_b, stride_gamma_h, stride_gamma_n,
    stride_state_u, stride_state_in_b, stride_state_in_h, stride_state_in_j, stride_state_in_d,
    stride_y_b, stride_y_h, stride_y_n, stride_y_d
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offsets_j = tl.arange(0, j_padded)
    offsets_d = tl.arange(0, d_head_padded)
    offsets_c = tl.arange(0, chunk_size)

    mask_j = offsets_j < j
    mask_d = offsets_d < d_head

    b_h_q_ptr = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    b_h_k_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    b_h_v_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h
    b_h_gamma_ptr = gamma_ptr + pid_b * stride_gamma_b + pid_h * stride_gamma_h
    b_h_y_ptr = Y_ptr + pid_b * stride_y_b + pid_h * stride_y_h

    b_h_state_in_ptr = states_in_ptr + pid_b * stride_state_in_b + pid_h * stride_state_in_h

    U = (N + chunk_size - 1) // chunk_size

    S = tl.zeros([j_padded, d_head_padded], dtype=tl.float32)

    for u in range(U):
        start = u * chunk_size
        t_offs = start + offsets_c
        mask_t = t_offs < N

        # Store S to states_in for backward pass
        state_in_ptrs = b_h_state_in_ptr + u * stride_state_u + offsets_j[:, None] * stride_state_in_j + offsets_d[None, :] * stride_state_in_d
        tl.store(state_in_ptrs, S, mask=mask_j[:, None] & mask_d[None, :])

        # Load Q_c, K_c, V_c, gamma_c
        q_ptrs = b_h_q_ptr + t_offs[:, None] * stride_q_n + offsets_j[None, :] * stride_q_j
        Q_c = tl.load(q_ptrs, mask=mask_t[:, None] & mask_j[None, :], other=0.0).to(tl.float32)

        k_ptrs = b_h_k_ptr + t_offs[:, None] * stride_k_n + offsets_j[None, :] * stride_k_j
        K_c = tl.load(k_ptrs, mask=mask_t[:, None] & mask_j[None, :], other=0.0).to(tl.float32)

        v_ptrs = b_h_v_ptr + t_offs[:, None] * stride_v_n + offsets_d[None, :] * stride_v_d
        V_c = tl.load(v_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        gamma_ptrs = b_h_gamma_ptr + t_offs * stride_gamma_n
        gamma_c = tl.load(gamma_ptrs, mask=mask_t, other=1.0).to(tl.float32)

        # 1. Parallel Cumulative Decay
        g_c = tl.log(gamma_c)
        g_cumsum = tl.cumsum(g_c, axis=0)
        g_ex = g_cumsum - g_c

        # 2. Parallel Intra-Chunk Attention Mask
        diff = g_ex[:, None] - g_cumsum[None, :]
        mask = tl.exp(diff) * tl.where(offsets_c[:, None] > offsets_c[None, :], 1.0, 0.0)

        # 3. Parallel Output Computation
        Q_c_f16 = Q_c.to(tl.float16)
        K_c_f16 = K_c.to(tl.float16)
        V_c_f16 = V_c.to(tl.float16)

        attn = tl.dot(Q_c_f16, tl.trans(K_c_f16))
        attn = (attn * mask).to(tl.float16)
        Y_intra = tl.dot(attn, V_c_f16)

        # 4. Parallel State Application
        S_f16 = S.to(tl.float16)
        Y_inter = tl.dot(Q_c_f16, S_f16) * tl.exp(g_ex)[:, None]
        Y_total = Y_intra + Y_inter

        # Store Y
        y_ptrs = b_h_y_ptr + t_offs[:, None] * stride_y_n + offsets_d[None, :] * stride_y_d
        tl.store(y_ptrs, Y_total.to(Y_ptr.dtype.element_ty), mask=mask_t[:, None] & mask_d[None, :])

        # 5. Parallel State Update
        last_idx = tl.max(tl.where(mask_t, offsets_c, 0))
        g_cumsum_last = tl.max(tl.where(offsets_c == last_idx, g_cumsum, -float('inf')))

        k_decay = tl.exp(g_cumsum_last - g_cumsum)
        K_c_decayed = K_c * k_decay[:, None] * tl.where(mask_t[:, None], 1.0, 0.0)
        K_c_decayed_f16 = K_c_decayed.to(tl.float16)

        S_update = tl.dot(tl.trans(K_c_decayed_f16), V_c_f16)
        S = S * tl.exp(g_cumsum_last) + S_update


@triton.jit
def chunk_gla_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, gamma_ptr,
    states_in_ptr, dY_ptr, 
    dQ_ptr, dK_ptr, dV_ptr, dgamma_ptr,
    dstate_ptr,              # (B, H, j, d_head) scratchpad
    N, chunk_size: tl.constexpr, 
    j, j_padded: tl.constexpr, d_head, d_head_padded: tl.constexpr,
    stride_q_b, stride_q_h, stride_q_n, stride_q_j,
    stride_k_b, stride_k_h, stride_k_n, stride_k_j,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_gamma_b, stride_gamma_h, stride_gamma_n,
    stride_state_u, stride_state_in_b, stride_state_in_h, stride_state_in_j, stride_state_in_d,
    stride_dy_b, stride_dy_h, stride_dy_n, stride_dy_d,
    stride_dq_b, stride_dq_h, stride_dq_n, stride_dq_j,
    stride_dk_b, stride_dk_h, stride_dk_n, stride_dk_j,
    stride_dv_b, stride_dv_h, stride_dv_n, stride_dv_d,
    stride_dgamma_b, stride_dgamma_h, stride_dgamma_n,
    stride_dstate_b, stride_dstate_h, stride_dstate_j, stride_dstate_d
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offsets_j = tl.arange(0, j_padded)
    state_mask = offsets_j < j

    b_h_q_ptr = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    b_h_k_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    b_h_v_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h
    b_h_gamma_ptr = gamma_ptr + pid_b * stride_gamma_b + pid_h * stride_gamma_h
    b_h_dy_ptr = dY_ptr + pid_b * stride_dy_b + pid_h * stride_dy_h

    b_h_state_in_ptr = states_in_ptr + pid_b * stride_state_in_b + pid_h * stride_state_in_h
    b_h_dstate_ptr = dstate_ptr + pid_b * stride_dstate_b + pid_h * stride_dstate_h

    U = (N + chunk_size - 1) // chunk_size

    for u in range(U - 1, -1, -1):
        start = u * chunk_size

        # Keeping the sequential backward loop to ensure gradients are mathematically correct
        for pid_d in range(d_head):
            dstate_ptrs = b_h_dstate_ptr + offsets_j * stride_dstate_j + pid_d * stride_dstate_d
            dstate_col = tl.load(dstate_ptrs, mask=state_mask, other=0.0)

            state_in_col_ptrs = b_h_state_in_ptr + u * stride_state_u + offsets_j * stride_state_in_j + pid_d * stride_state_in_d
            S_cur = tl.load(state_in_col_ptrs, mask=state_mask, other=0.0)

            S_prev_arr = tl.zeros((chunk_size, j_padded), dtype=tl.float32)
            
            for t_in_chunk in range(chunk_size):
                t = start + t_in_chunk
                if t < N:
                    mask_t = (tl.arange(0, chunk_size)[:, None] == t_in_chunk)
                    S_prev_arr = tl.where(mask_t, S_cur[None, :], S_prev_arr)

                    k_t_ptrs = b_h_k_ptr + t * stride_k_n + offsets_j * stride_k_j
                    k_t = tl.load(k_t_ptrs, mask=state_mask, other=0.0).to(tl.float32)
                    
                    v_t_ptr = b_h_v_ptr + t * stride_v_n + pid_d * stride_v_d
                    v_t = tl.load(v_t_ptr).to(tl.float32)
                    
                    gamma_t_ptr = b_h_gamma_ptr + t * stride_gamma_n
                    gamma_t = tl.load(gamma_t_ptr).to(tl.float32)
                    
                    S_cur = gamma_t * S_cur + k_t * v_t

            for i in range(chunk_size):
                t_in_chunk = chunk_size - 1 - i
                t = start + t_in_chunk
                if t < N:
                    mask_t = (tl.arange(0, chunk_size)[:, None] == t_in_chunk)
                    S_prev = tl.sum(tl.where(mask_t, S_prev_arr, 0.0), axis=0)

                    q_t_ptrs = b_h_q_ptr + t * stride_q_n + offsets_j * stride_q_j
                    k_t_ptrs = b_h_k_ptr + t * stride_k_n + offsets_j * stride_k_j
                    q_t = tl.load(q_t_ptrs, mask=state_mask, other=0.0).to(tl.float32)
                    k_t = tl.load(k_t_ptrs, mask=state_mask, other=0.0).to(tl.float32)

                    v_t_ptr = b_h_v_ptr + t * stride_v_n + pid_d * stride_v_d
                    v_t = tl.load(v_t_ptr).to(tl.float32)

                    gamma_t_ptr = b_h_gamma_ptr + t * stride_gamma_n
                    gamma_t = tl.load(gamma_t_ptr).to(tl.float32)

                    dY_t_ptr = b_h_dy_ptr + t * stride_dy_n + pid_d * stride_dy_d
                    dY_t = tl.load(dY_t_ptr).to(tl.float32)

                    dq_val = dY_t * S_prev
                    dk_val = dstate_col * v_t
                    dv_val = tl.sum(dstate_col * k_t)
                    dgamma_val = tl.sum(dstate_col * S_prev)

                    dq_t_ptrs = dQ_ptr + pid_b * stride_dq_b + pid_h * stride_dq_h + t * stride_dq_n + offsets_j * stride_dq_j
                    dk_t_ptrs = dK_ptr + pid_b * stride_dk_b + pid_h * stride_dk_h + t * stride_dk_n + offsets_j * stride_dk_j
                    dv_t_ptr = dV_ptr + pid_b * stride_dv_b + pid_h * stride_dv_h + t * stride_dv_n + pid_d * stride_dv_d
                    dgamma_t_ptr = dgamma_ptr + pid_b * stride_dgamma_b + pid_h * stride_dgamma_h + t * stride_dgamma_n

                    tl.atomic_add(dq_t_ptrs, dq_val, mask=state_mask)
                    tl.atomic_add(dk_t_ptrs, dk_val, mask=state_mask)
                    tl.atomic_add(dgamma_t_ptr, dgamma_val)
                    tl.store(dv_t_ptr, dv_val.to(dV_ptr.dtype.element_ty))

                    dstate_col = gamma_t * dstate_col + dY_t * q_t
            
            tl.store(dstate_ptrs, dstate_col, mask=state_mask)


class ChunkGLAInlier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, gamma, chunk_size=32):
        chunk_size = 64 # Override to 64 for optimal parallel block performance

        B, H, N, j = Q.shape
        d_head = V.shape[-1]
        device = Q.device
        
        j_padded = 1 << (j - 1).bit_length()
        j_padded = max(j_padded, 16)

        d_head_padded = 1 << (d_head - 1).bit_length()
        d_head_padded = max(d_head_padded, 16)
        
        Y = torch.empty(B, H, N, d_head, dtype=Q.dtype, device=device)
        U = (N + chunk_size - 1) // chunk_size
        states_in = torch.empty(U, B, H, j, d_head, dtype=torch.float32, device=device)
        
        grid = (B, H)
        
        chunk_gla_fwd_kernel[grid](
            Q, K, V, gamma,
            states_in, Y,
            N, chunk_size, j, j_padded, d_head, d_head_padded,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            gamma.stride(0), gamma.stride(1), gamma.stride(2),
            states_in.stride(0), states_in.stride(1), states_in.stride(2), states_in.stride(3), states_in.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3)
        )
            
        ctx.save_for_backward(Q, K, V, gamma, states_in)
        ctx.chunk_size = chunk_size
        ctx.j_padded = j_padded
        ctx.d_head_padded = d_head_padded
        return Y

    @staticmethod
    def backward(ctx, dY):
        Q, K, V, gamma, states_in = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        j_padded = ctx.j_padded
        d_head_padded = ctx.d_head_padded
        
        B, H, N, j = Q.shape
        d_head = V.shape[-1]
        device = Q.device
        
        dQ = torch.zeros(B, H, N, j, dtype=torch.float32, device=device)
        dK = torch.zeros(B, H, N, j, dtype=torch.float32, device=device)
        dgamma = torch.zeros(B, H, N, dtype=torch.float32, device=device)
        dV = torch.zeros(B, H, N, d_head, dtype=Q.dtype, device=device)
        dstate = torch.zeros(B, H, j, d_head, dtype=torch.float32, device=device)
        
        grid = (B, H)
        
        chunk_gla_bwd_kernel[grid](
            Q, K, V, gamma,
            states_in, dY,
            dQ, dK, dV, dgamma, dstate,
            N, chunk_size, j, j_padded, d_head, d_head_padded,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            gamma.stride(0), gamma.stride(1), gamma.stride(2),
            states_in.stride(0), states_in.stride(1), states_in.stride(2), states_in.stride(3), states_in.stride(4),
            dY.stride(0), dY.stride(1), dY.stride(2), dY.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            dgamma.stride(0), dgamma.stride(1), dgamma.stride(2),
            dstate.stride(0), dstate.stride(1), dstate.stride(2), dstate.stride(3)
        )
            
        return dQ.to(Q.dtype), dK.to(K.dtype), dV, dgamma.to(gamma.dtype), None
