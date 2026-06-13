import torch
import triton
import triton.language as tl

def triton_next_power_of_2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

@triton.jit
def _fused_hofa_decode_kernel(
    Q_ptr, K_ptr, V_ptr,
    K_cache_ptr, V_cache_ptr,
    State_I_ptr,
    Outlier_idx_ptr, Inlier_idx_ptr,
    Gate_W_ptr, Gate_B_ptr,
    Y_out_ptr,
    seq_len, alpha,
    stride_b, stride_h, stride_d,
    cache_stride_b, cache_stride_h, cache_stride_sl, cache_stride_d,
    state_stride_b, state_stride_h, state_stride_j, state_stride_d,
    gw_stride_h, gw_stride_d,
    D_HEAD: tl.constexpr, R: tl.constexpr, J: tl.constexpr,
    R_PAD: tl.constexpr, J_PAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr
):
    # Grid: (B, H)
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Offsets
    q_offset = batch_idx * stride_b + head_idx * stride_h
    state_offset = batch_idx * state_stride_b + head_idx * state_stride_h
    gw_offset = head_idx * gw_stride_h

    # Load routing indices with padding masks
    offs_r = tl.arange(0, R_PAD)
    mask_r = offs_r < R
    outlier_idx = tl.load(Outlier_idx_ptr + head_idx * R + offs_r, mask=mask_r, other=0)
    
    offs_j = tl.arange(0, J_PAD)
    mask_j = offs_j < J
    inlier_idx = tl.load(Inlier_idx_ptr + head_idx * J + offs_j, mask=mask_j, other=0)

    # Load Q, K, V for current token
    offs_d = tl.arange(0, D_HEAD)
    q = tl.load(Q_ptr + q_offset + offs_d)
    k = tl.load(K_ptr + q_offset + offs_d)
    v = tl.load(V_ptr + q_offset + offs_d)

    # --- Compute Gate ---
    gate_w_q = tl.load(Gate_W_ptr + gw_offset + offs_d)
    gate_w_k = tl.load(Gate_W_ptr + gw_offset + D_HEAD + offs_d)
    gate_b = tl.load(Gate_B_ptr + head_idx)
    
    gate_logit = tl.sum(q * gate_w_q) + tl.sum(k * gate_w_k) + gate_b
    gate_logit = gate_logit.to(tl.float32)
    gamma = 1.0 / (1.0 + tl.math.exp(gate_logit))
    gamma = gamma.to(q.dtype)

    # --- Inlier State Update ---
    offs_j_2d = tl.arange(0, J_PAD)[:, None]
    offs_d_2d = tl.arange(0, D_HEAD)[None, :]
    mask_j_2d = offs_j_2d < J
    
    state_ptrs = State_I_ptr + state_offset + offs_j_2d * state_stride_j + offs_d_2d * state_stride_d
    state = tl.load(state_ptrs, mask=mask_j_2d, other=0.0)

    # Gather Q_J and K_J using indices
    q_J = tl.load(Q_ptr + q_offset + inlier_idx, mask=mask_j, other=0.0)
    k_J = tl.load(K_ptr + q_offset + inlier_idx, mask=mask_j, other=0.0)

    # Y_I = sum_j (q_J[j] * state[j, d])
    y_i = tl.sum(q_J[:, None] * state, axis=0)

    # state_new = gamma * state + k_J[j] * v[d]
    state_new = gamma * state + k_J[:, None] * v[None, :]
    tl.store(state_ptrs, state_new, mask=mask_j_2d)

    # --- Outlier Exact Attention ---
    q_O = tl.load(Q_ptr + q_offset + outlier_idx, mask=mask_r, other=0.0)
    
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D_HEAD], dtype=tl.float32)
    
    cache_base_k = K_cache_ptr + batch_idx * cache_stride_b + head_idx * cache_stride_h
    cache_base_v = V_cache_ptr + batch_idx * cache_stride_b + head_idx * cache_stride_h
    
    for start_n in range(0, seq_len, BLOCK_SEQ):
        offs_n = start_n + tl.arange(0, BLOCK_SEQ)
        mask_n = offs_n < seq_len
        
        # k_O block: (BLOCK_SEQ, R_PAD)
        k_ptrs = cache_base_k + offs_n[:, None] * cache_stride_sl + outlier_idx[None, :] * cache_stride_d
        mask_k = mask_n[:, None] & mask_r[None, :]
        k_O_block = tl.load(k_ptrs, mask=mask_k, other=0.0)
        
        # dot product over R_PAD
        qk = tl.sum(q_O[None, :] * k_O_block, axis=1).to(tl.float32)
        qk = tl.where(mask_n, qk, -float("inf"))
        
        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(qk))
        p = tl.math.exp(qk - m_ij)
        l_ij = tl.sum(p)
        
        alpha_sm = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha_sm + l_ij
        
        # v_block: (BLOCK_SEQ, D_HEAD)
        v_ptrs = cache_base_v + offs_n[:, None] * cache_stride_sl + offs_d_2d * cache_stride_d
        v_block = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        acc = acc * alpha_sm + tl.sum(p[:, None].to(v_block.dtype) * v_block, axis=0)
        m_i = m_ij

    y_o = acc / l_i
    y_o = y_o.to(q.dtype)

    # --- Blend and Store ---
    y_out = alpha * y_o + (1.0 - alpha) * y_i
    y_out_ptrs = Y_out_ptr + q_offset + offs_d
    tl.store(y_out_ptrs, y_out)


def fused_hofa_decode(Q, K, V, K_cache, V_cache, state_I, outlier_idx, inlier_idx, gate_weight, gate_bias, alpha, seq_len):
    """
    Q, K, V: (B, H, 1, d_head)
    K_cache, V_cache: (B, H, max_sl, d_head)
    state_I: (B, H, j, d_head)
    outlier_idx: (H, r) or (r,)
    inlier_idx: (H, j) or (j,)
    gate_weight: (H, 2*d_head)
    gate_bias: (H)
    """
    B, H, N, D_HEAD = Q.shape
    assert N == 1, "fused_hofa_decode only supports step-by-step decoding (N=1)"
    
    if outlier_idx.dim() == 1:
        outlier_idx = outlier_idx.unsqueeze(0).expand(H, -1).contiguous()
    if inlier_idx.dim() == 1:
        inlier_idx = inlier_idx.unsqueeze(0).expand(H, -1).contiguous()
        
    R = outlier_idx.shape[1]
    J = inlier_idx.shape[1]
    
    Y_out = torch.empty_like(Q)
    
    grid = (B, H)
    BLOCK_SEQ = 128
    
    R_PAD = triton_next_power_of_2(R) if R > 0 else 16
    J_PAD = triton_next_power_of_2(J) if J > 0 else 16
    
    _fused_hofa_decode_kernel[grid](
        Q, K, V,
        K_cache, V_cache,
        state_I,
        outlier_idx, inlier_idx,
        gate_weight, gate_bias,
        Y_out,
        seq_len, alpha.item(),
        Q.stride(0), Q.stride(1), Q.stride(3),
        K_cache.stride(0), K_cache.stride(1), K_cache.stride(2), K_cache.stride(3),
        state_I.stride(0), state_I.stride(1), state_I.stride(2), state_I.stride(3),
        gate_weight.stride(0), gate_weight.stride(1),
        D_HEAD=D_HEAD, R=R, J=J,
        R_PAD=R_PAD, J_PAD=J_PAD,
        BLOCK_SEQ=BLOCK_SEQ
    )
    
    return Y_out, state_I
