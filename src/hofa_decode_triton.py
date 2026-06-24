import torch
import triton
import triton.language as tl

def triton_next_power_of_2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

@triton.jit
def _fused_hofa_decode_kernel(
    Q, K, V, 
    Cache_K, Cache_V, 
    State_I,
    Log_Gamma, Mix_G,
    Norm_W,
    Y, State_I_new,
    R, seq_len, sm_scale,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    cache_stride_b, cache_stride_h, cache_stride_sl, cache_stride_d,
    state_stride_b, state_stride_h, state_stride_j, state_stride_d,
    stride_lg_b, stride_lg_h, stride_lg_n,
    stride_mg_b, stride_mg_h, stride_mg_n,
    stride_nw_h, stride_nw_d,
    stride_y_b, stride_y_h, stride_y_n, stride_y_d,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    D_HEAD: tl.constexpr,
    R_PAD: tl.constexpr,
    J: tl.constexpr,
    J_PAD: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_d = tl.arange(0, D_HEAD)
    
    q_ptrs = Q + batch_idx * stride_q_b + head_idx * stride_q_h + offs_d * stride_q_d
    k_step_ptrs = K + batch_idx * stride_k_b + head_idx * stride_k_h + offs_d * stride_k_d
    v_step_ptrs = V + batch_idx * stride_v_b + head_idx * stride_v_h + offs_d * stride_v_d

    q = tl.load(q_ptrs)
    k_step = tl.load(k_step_ptrs)
    v_step = tl.load(v_step_ptrs)

    lg_ptr = Log_Gamma + batch_idx * stride_lg_b + head_idx * stride_lg_h
    mg_ptr = Mix_G + batch_idx * stride_mg_b + head_idx * stride_mg_h
    log_gamma = tl.load(lg_ptr).to(tl.float32)
    mix_g = tl.load(mg_ptr)
    gamma = tl.math.exp(log_gamma)

    offs_j_2d = tl.arange(0, J_PAD)[:, None]
    offs_d_2d = tl.arange(0, D_HEAD)[None, :]
    mask_j_2d = offs_j_2d < J
    
    state_ptrs = State_I + batch_idx * state_stride_b + head_idx * state_stride_h + offs_j_2d * state_stride_j + offs_d_2d * state_stride_d
    state = tl.load(state_ptrs, mask=mask_j_2d, other=0.0)

    offs_j = R + tl.arange(0, J_PAD)
    mask_j = (offs_j - R) < J
    
    k_J = tl.load(K + batch_idx * stride_k_b + head_idx * stride_k_h + offs_j * stride_k_d, mask=mask_j, other=0.0)
    q_J = tl.load(Q + batch_idx * stride_q_b + head_idx * stride_q_h + offs_j * stride_q_d, mask=mask_j, other=0.0)

    y_i = tl.sum(q_J[:, None] * state, axis=0)

    state_new = gamma * state + k_J[:, None] * v_step[None, :]
    tl.store(state_ptrs, state_new, mask=mask_j_2d)

    cache_base_k = Cache_K + batch_idx * cache_stride_b + head_idx * cache_stride_h
    cache_base_v = Cache_V + batch_idx * cache_stride_b + head_idx * cache_stride_h

    m_i = tl.full([1], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc_O = tl.zeros([D_HEAD], dtype=tl.float32)

    offs_r = tl.arange(0, R_PAD)
    mask_r = offs_r < R
    q_base = Q + batch_idx * stride_q_b + head_idx * stride_q_h
    q_O = tl.load(q_base + offs_r * stride_q_d, mask=mask_r, other=0.0)

    for start_n in range(0, seq_len, BLOCK_SEQ):
        offs_n = start_n + tl.arange(0, BLOCK_SEQ)
        mask_n = offs_n < seq_len

        k_ptrs = cache_base_k + offs_n[:, None] * cache_stride_sl + offs_r[None, :] * cache_stride_d
        v_ptrs = cache_base_v + offs_n[:, None] * cache_stride_sl + offs_d_2d * cache_stride_d
        
        k_O_block = tl.load(k_ptrs, mask=mask_n[:, None] & mask_r[None, :], other=0.0)
        v_block = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        attn_scores = tl.sum(q_O[None, :] * k_O_block, axis=1) * sm_scale
        attn_scores = tl.where(mask_n, attn_scores, float('-inf'))

        m_ij = tl.maximum(m_i, tl.max(attn_scores, axis=0))
        p_ij = tl.math.exp(attn_scores - m_ij)
        l_ij = tl.sum(p_ij, axis=0)

        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc_O = acc_O * alpha + tl.sum(v_block * p_ij[:, None], axis=0)
        m_i = m_ij

    y_o = acc_O / l_i

    var = tl.sum(y_i * y_i, axis=0) / D_HEAD
    rsqrt = tl.math.rsqrt(var + 1e-5)
    norm_w = tl.load(Norm_W + head_idx * stride_nw_h + offs_d * stride_nw_d)
    y_i_norm = y_i * rsqrt * norm_w

    y_out = mix_g * y_o + (1.0 - mix_g) * y_i_norm
    tl.store(Y + batch_idx * stride_y_b + head_idx * stride_y_h + offs_d * stride_y_d, y_out)

def fused_hofa_decode(
    q, k, v, 
    cache_k, cache_v, 
    state_I, 
    log_gamma, mix_g,
    norm_w,
    R, seq_len, sm_scale
):
    B, H, N, D = q.shape
    assert N == 1

    Y = torch.empty_like(q)
    state_I_new = torch.empty_like(state_I)

    BLOCK_HEADS = 1
    BLOCK_SEQ = 128
    D_HEAD = triton.next_power_of_2(D)
    R_PAD = triton.next_power_of_2(R)
    J = D - R
    J_PAD = triton.next_power_of_2(J)

    grid = (B, H)

    _fused_hofa_decode_kernel[grid](
        q, k, v,
        cache_k, cache_v,
        state_I,
        log_gamma, mix_g,
        norm_w,
        Y, state_I_new,
        R, seq_len, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        cache_k.stride(0), cache_k.stride(1), cache_k.stride(2), cache_k.stride(3),
        state_I.stride(0), state_I.stride(1), state_I.stride(2), state_I.stride(3),
        log_gamma.stride(0), log_gamma.stride(1), log_gamma.stride(2),
        mix_g.stride(0), mix_g.stride(1), mix_g.stride(2),
        norm_w.stride(0), norm_w.stride(1),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        BLOCK_HEADS=BLOCK_HEADS,
        BLOCK_SEQ=BLOCK_SEQ,
        D_HEAD=D_HEAD,
        R_PAD=R_PAD,
        J=J,
        J_PAD=J_PAD
    )

    return Y, state_I_new
