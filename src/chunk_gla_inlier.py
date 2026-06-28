import torch
import triton
import triton.language as tl
import warnings
import torch.nn.functional as F

@triton.jit
def chunk_gla_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, log_gamma_ptr,
    states_in_ptr,           
    Y_ptr,                   
    N, chunk_size: tl.constexpr, 
    r: tl.constexpr, j, j_padded: tl.constexpr, d_head, d_head_padded: tl.constexpr,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_log_gamma_b, stride_log_gamma_h, stride_log_gamma_n,
    stride_state_u, stride_state_in_b, stride_state_in_h, stride_state_in_j, stride_state_in_d,
    stride_y_b, stride_y_h, stride_y_n, stride_y_d,
    REQUIRES_GRAD: tl.constexpr
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
        Q_c = (tl.load(q_ptrs, mask=mask_t[:, None] & mask_j[None, :], other=0.0)).to(Q_ptr.dtype.element_ty)

        k_ptrs = b_h_k_ptr + t_offs[:, None] * stride_k_n + inlier_indices[None, :] * stride_k_d
        K_c = tl.load(k_ptrs, mask=mask_t[:, None] & mask_j[None, :], other=0.0)

        v_ptrs = b_h_v_ptr + t_offs[:, None] * stride_v_n + offsets_d[None, :] * stride_v_d
        V_c = tl.load(v_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0)

        log_gamma_ptrs = b_h_log_gamma_ptr + t_offs * stride_log_gamma_n
        g_c = tl.load(log_gamma_ptrs, mask=mask_t, other=0.0).to(tl.float32)

        g_cumsum = tl.cumsum(g_c, axis=0)

        diff = g_cumsum[:, None] - g_cumsum[None, :]
        diff = tl.where(offsets_c[:, None] >= offsets_c[None, :], diff, -float('inf'))
        mask = tl.math.exp(diff)



        attn = tl.dot(Q_c, tl.trans(K_c), allow_tf32=False)
        attn = (attn * mask).to(Q_c.dtype)
        Y_intra = tl.dot(attn, V_c, allow_tf32=False)

        Y_inter = tl.dot(Q_c, S.to(Q_c.dtype), allow_tf32=False) * tl.math.exp(g_cumsum)[:, None]
        Y_total = Y_intra + Y_inter

        y_ptrs = b_h_y_ptr + t_offs[:, None] * stride_y_n + offsets_d[None, :] * stride_y_d
        tl.store(y_ptrs, Y_total.to(Y_ptr.dtype.element_ty), mask=mask_t[:, None] & mask_d[None, :])

        last_idx = tl.max(tl.where(mask_t, offsets_c, 0))
        g_cumsum_last = tl.max(tl.where(offsets_c == last_idx, g_cumsum, -float('inf')))

        k_decay = tl.math.exp(g_cumsum_last - g_cumsum)
        K_c_decayed = K_c * k_decay[:, None] * tl.where(mask_t[:, None], 1.0, 0.0)
        
        S_update = tl.dot(tl.trans(K_c_decayed.to(Q_c.dtype)), V_c, allow_tf32=False)
        S = S * tl.math.exp(g_cumsum_last) + S_update


class ChunkGLAInlier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, log_gamma, r, chunk_size):
        # Q, K, V: (B, H, N, d_head)
        # log_gamma: (B, H, N)
        # r: scalar
        # chunk_size: scalar
        B, H, N, d_head = Q.shape
        j = d_head - r
        device = Q.device
        
        j_padded = 1 << (j - 1).bit_length()
        j_padded = max(j_padded, 16)
        d_head_padded = 1 << (d_head - 1).bit_length()
        d_head_padded = max(d_head_padded, 16)

        Y = torch.zeros(B, H, N, d_head, dtype=Q.dtype, device=device)
        U = (N + chunk_size - 1) // chunk_size
        
        requires_grad = Q.requires_grad or K.requires_grad or V.requires_grad or log_gamma.requires_grad
        
        if requires_grad:
            states_in = torch.empty(B, H, U, j, d_head, device=device, dtype=torch.float32)
        else:
            states_in = torch.empty(1, 1, 1, 1, 1, device=device, dtype=torch.float32)
            
        grid = (B, H)
        
        chunk_gla_fwd_kernel[grid](
            Q, K, V, log_gamma,
            states_in,
            Y,
            N, chunk_size,
            r, j, j_padded, d_head, d_head_padded,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            log_gamma.stride(0), log_gamma.stride(1), log_gamma.stride(2),
            states_in.stride(2), states_in.stride(0), states_in.stride(1), states_in.stride(3), states_in.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            REQUIRES_GRAD=requires_grad
        )
            
        if requires_grad:
            ctx.save_for_backward(Q, K, V, log_gamma, states_in)
            ctx.r = r
            ctx.j = j
            ctx.d_head = d_head
        
        return Y

    @staticmethod
    def backward(ctx, dY):
        try:
            from fla.ops.gla import chunk_gla
        except ImportError:
            raise ImportError("fla library is required for the exact backward pass.")
            
        Q, K, V, log_gamma, states_in = ctx.saved_tensors
        r = ctx.r
        B, H, N, D = Q.shape
        j = ctx.j
        d_head = ctx.d_head
        
        # Gather the inlier features
        inlier_indices = torch.arange(r, r + j, device=Q.device)
        Q_J = Q.index_select(-1, inlier_indices)
        K_J = K.index_select(-1, inlier_indices)
        
        # Pad Q_J, K_J along the feature dimension to match d_head
        pad_size = d_head - j
        if pad_size > 0:
            Q_J = F.pad(Q_J, (0, pad_size))
            K_J = F.pad(K_J, (0, pad_size))

        log_gamma_pad = log_gamma.unsqueeze(-1).expand(B, H, N, d_head)

        Q_fla = Q_J.transpose(1, 2).contiguous()
        K_fla = K_J.transpose(1, 2).contiguous()
        V_fla = V.transpose(1, 2).contiguous()
        g_fla = log_gamma_pad.transpose(1, 2).contiguous()
        
        Q_fla.requires_grad_(True)
        K_fla.requires_grad_(True)
        V_fla.requires_grad_(True)
        g_fla.requires_grad_(True)
        
        with torch.enable_grad():
            Y_fla, v_new_fla = chunk_gla(Q_fla, K_fla, V_fla, g_fla, scale=1.0)
            dY_fla = dY.transpose(1, 2).contiguous()
            Y_fla.backward(dY_fla)
        
        dQ_J = Q_fla.grad.transpose(1, 2)
        dK_J = K_fla.grad.transpose(1, 2)
        dV = V_fla.grad.transpose(1, 2)
        dlog_gamma_pad = g_fla.grad.transpose(1, 2)
        
        if pad_size > 0:
            dQ_J = dQ_J[..., :j]
            dK_J = dK_J[..., :j]
            dlog_gamma = dlog_gamma_pad.sum(dim=-1)
        else:
            dlog_gamma = dlog_gamma_pad
            
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dQ.scatter_add_(-1, inlier_indices.view(1, 1, 1, j).expand(B, H, N, j), dQ_J)
        dK.scatter_add_(-1, inlier_indices.view(1, 1, 1, j).expand(B, H, N, j), dK_J)
        
        return dQ, dK, dV, dlog_gamma, None, None
