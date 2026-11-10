import torch
import triton
import triton.language as tl
from src.modules.triton_utils import get_bucket

def triton_next_power_of_2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

_SM_COUNT_CACHE = {}
_SPLIT_WORKSPACE_CACHE = {}

def _get_cached_sm_count(device):
    dev_idx = device.index if device.index is not None else 0
    if dev_idx not in _SM_COUNT_CACHE:
        _SM_COUNT_CACHE[dev_idx] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SM_COUNT_CACHE[dev_idx]

def _get_cached_split_workspace(device, B, H, max_splits, D_HEAD):
    key = (device.index, B, H, max_splits, D_HEAD)
    if key not in _SPLIT_WORKSPACE_CACHE:
        mid_o = torch.empty((B, H, max_splits, D_HEAD), dtype=torch.float32, device=device)
        mid_l = torch.empty((B, H, max_splits), dtype=torch.float32, device=device)
        mid_m = torch.empty((B, H, max_splits), dtype=torch.float32, device=device)
        _SPLIT_WORKSPACE_CACHE[key] = (mid_o, mid_l, mid_m)
    return _SPLIT_WORKSPACE_CACHE[key]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SEQ': 16}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_SEQ': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SEQ': 32}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_SEQ': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SEQ': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SEQ': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SEQ': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SEQ': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SEQ': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SEQ': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SEQ': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SEQ': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SEQ': 256}, num_warps=8, num_stages=4),
    ],
    key=['seq_len_bucket']
)
@triton.jit
def _fused_hofa_decode_kernel(
    Q, K, V, 
    Cache_K, Cache_V, 
    State_I,
    Log_Gamma, Mix_G,
    Norm_W,
    Log_Alpha,
    Y, State_I_new,
    R, seq_len, seq_len_bucket, sm_scale, eps,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    cache_k_stride_b, cache_k_stride_h, cache_k_stride_sl, cache_k_stride_d,
    cache_v_stride_b, cache_v_stride_h, cache_v_stride_sl, cache_v_stride_d,
    state_stride_b, state_stride_h, state_stride_j, state_stride_d,
    stride_lg_b, stride_lg_h, stride_lg_n,
    stride_mg_b, stride_mg_h, stride_mg_n,
    stride_nw_h, stride_nw_d,
    stride_la_b, stride_la_h, stride_la_j,
    stride_y_b, stride_y_h, stride_y_n, stride_y_d,
    HAS_ALPHA: tl.constexpr,
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
    
    v_step_ptrs = V + batch_idx * stride_v_b + head_idx * stride_v_h + offs_d * stride_v_d
    v_step = tl.load(v_step_ptrs).to(tl.float32)

    lg_ptr = Log_Gamma + batch_idx * stride_lg_b + head_idx * stride_lg_h
    mg_ptr = Mix_G + batch_idx * stride_mg_b + head_idx * stride_mg_h
    log_gamma = tl.load(lg_ptr).to(tl.float32)
    mix_g = tl.load(mg_ptr).to(tl.float32)
    gamma = tl.math.exp(log_gamma)

    offs_d_2d = tl.arange(0, D_HEAD)[None, :]
    y_i = tl.zeros([D_HEAD], dtype=tl.float32)
    BLOCK_J: tl.constexpr = 16
    for start_j in range(0, J, BLOCK_J):
        offs_j_chunk = start_j + tl.arange(0, BLOCK_J)
        mask_j_chunk = offs_j_chunk < J
        
        state_ptrs = State_I + batch_idx * state_stride_b + head_idx * state_stride_h + offs_j_chunk[:, None] * state_stride_j + offs_d_2d * state_stride_d
        state_chunk = tl.load(state_ptrs, mask=mask_j_chunk[:, None], other=0.0).to(tl.float32)
        
        k_chunk = tl.load(K + batch_idx * stride_k_b + head_idx * stride_k_h + (R + offs_j_chunk) * stride_k_d, mask=mask_j_chunk, other=0.0).to(tl.float32)
        q_chunk = tl.load(Q + batch_idx * stride_q_b + head_idx * stride_q_h + (R + offs_j_chunk) * stride_q_d, mask=mask_j_chunk, other=0.0).to(tl.float32)
        
        if HAS_ALPHA:
            la_ptrs = Log_Alpha + batch_idx * stride_la_b + head_idx * stride_la_h + offs_j_chunk * stride_la_j
            alpha_chunk = tl.load(la_ptrs, mask=mask_j_chunk, other=0.0).to(tl.float32)
            gamma_chunk = tl.math.exp(log_gamma * tl.math.exp(alpha_chunk))[:, None]
        else:
            gamma_chunk = gamma

        state_new_chunk = gamma_chunk * state_chunk + k_chunk[:, None] * v_step[None, :]
        y_i += tl.sum(q_chunk[:, None] * state_new_chunk, axis=0)
        
        state_new_ptrs = State_I_new + batch_idx * state_stride_b + head_idx * state_stride_h + offs_j_chunk[:, None] * state_stride_j + offs_d_2d * state_stride_d
        tl.store(state_new_ptrs, state_new_chunk, mask=mask_j_chunk[:, None])

    cache_base_k = Cache_K + batch_idx * cache_k_stride_b + head_idx * cache_k_stride_h
    cache_base_v = Cache_V + batch_idx * cache_v_stride_b + head_idx * cache_v_stride_h

    m_i = tl.full([1], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc_O = tl.zeros([D_HEAD], dtype=tl.float32)

    offs_r = tl.arange(0, R_PAD)
    mask_r = offs_r < R
    q_base = Q + batch_idx * stride_q_b + head_idx * stride_q_h
    q_O = tl.load(q_base + offs_r * stride_q_d, mask=mask_r, other=0.0).to(tl.float32)

    for start_n in range(0, seq_len, BLOCK_SEQ):
        offs_n = start_n + tl.arange(0, BLOCK_SEQ)
        mask_n = offs_n < seq_len

        k_ptrs = cache_base_k + offs_n[:, None] * cache_k_stride_sl + offs_r[None, :] * cache_k_stride_d
        v_ptrs = cache_base_v + offs_n[:, None] * cache_v_stride_sl + offs_d_2d * cache_v_stride_d
        
        k_O_block = tl.load(k_ptrs, mask=mask_n[:, None] & mask_r[None, :], other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

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
    rsqrt = tl.math.rsqrt(var + eps)
    norm_w = tl.load(Norm_W + head_idx * stride_nw_h + offs_d * stride_nw_d)
    y_i_norm = y_i * rsqrt * norm_w

    y_out = mix_g * y_o + (1.0 - mix_g) * y_i_norm
    tl.store(Y + batch_idx * stride_y_b + head_idx * stride_y_h + offs_d * stride_y_d, y_out.to(Y.dtype.element_ty))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SEQ': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SEQ': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SEQ': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SEQ': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SEQ': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SEQ': 256}, num_warps=8, num_stages=3),
    ],
    key=['tokens_per_split_bucket']
)
@triton.jit
def _fused_hofa_decode_split_kernel(
    Q, Cache_K, Cache_V,
    Mid_O, Mid_L, Mid_M,
    R, seq_len, tokens_per_split, tokens_per_split_bucket, sm_scale,
    split_k,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    cache_k_stride_b, cache_k_stride_h, cache_k_stride_sl, cache_k_stride_d,
    cache_v_stride_b, cache_v_stride_h, cache_v_stride_sl, cache_v_stride_d,
    mid_o_stride_b, mid_o_stride_h, mid_o_stride_s, mid_o_stride_d,
    mid_l_stride_b, mid_l_stride_h, mid_l_stride_s,
    mid_m_stride_b, mid_m_stride_h, mid_m_stride_s,
    BLOCK_SEQ: tl.constexpr,
    D_HEAD: tl.constexpr,
    R_PAD: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    start_tok = split_idx * tokens_per_split
    end_tok = tl.minimum(start_tok + tokens_per_split, seq_len)

    offs_d = tl.arange(0, D_HEAD)
    offs_r = tl.arange(0, R_PAD)
    mask_r = offs_r < R

    cache_base_k = Cache_K + batch_idx * cache_k_stride_b + head_idx * cache_k_stride_h
    cache_base_v = Cache_V + batch_idx * cache_v_stride_b + head_idx * cache_v_stride_h

    q_base = Q + batch_idx * stride_q_b + head_idx * stride_q_h
    q_O = tl.load(q_base + offs_r * stride_q_d, mask=mask_r, other=0.0).to(tl.float32)

    m_i = tl.full([1], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc_O = tl.zeros([D_HEAD], dtype=tl.float32)

    offs_d_2d = offs_d[None, :]

    for start_n in range(start_tok, end_tok, BLOCK_SEQ):
        offs_n = start_n + tl.arange(0, BLOCK_SEQ)
        mask_n = offs_n < end_tok

        k_ptrs = cache_base_k + offs_n[:, None] * cache_k_stride_sl + offs_r[None, :] * cache_k_stride_d
        v_ptrs = cache_base_v + offs_n[:, None] * cache_v_stride_sl + offs_d_2d * cache_v_stride_d

        k_O_block = tl.load(k_ptrs, mask=mask_n[:, None] & mask_r[None, :], other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        attn_scores = tl.sum(q_O[None, :] * k_O_block, axis=1) * sm_scale
        attn_scores = tl.where(mask_n, attn_scores, float('-inf'))

        m_ij = tl.maximum(m_i, tl.max(attn_scores, axis=0))
        p_ij = tl.math.exp(attn_scores - m_ij)
        l_ij = tl.sum(p_ij, axis=0)

        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc_O = acc_O * alpha + tl.sum(v_block * p_ij[:, None], axis=0)
        m_i = m_ij

    mid_m_ptr = Mid_M + batch_idx * mid_m_stride_b + head_idx * mid_m_stride_h + split_idx * mid_m_stride_s
    mid_l_ptr = Mid_L + batch_idx * mid_l_stride_b + head_idx * mid_l_stride_h + split_idx * mid_l_stride_s
    mid_o_ptrs = Mid_O + batch_idx * mid_o_stride_b + head_idx * mid_o_stride_h + split_idx * mid_o_stride_s + offs_d * mid_o_stride_d

    tl.store(mid_m_ptr + tl.arange(0, 1), m_i)
    tl.store(mid_l_ptr + tl.arange(0, 1), l_i)
    tl.store(mid_o_ptrs, acc_O)

@triton.jit
def _fused_hofa_decode_reduce_kernel(
    Q, K, V,
    Mid_O, Mid_L, Mid_M,
    State_I, State_I_new,
    Log_Gamma, Mix_G,
    Norm_W,
    Log_Alpha,
    Y,
    R, split_k, eps,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    mid_o_stride_b, mid_o_stride_h, mid_o_stride_s, mid_o_stride_d,
    mid_l_stride_b, mid_l_stride_h, mid_l_stride_s,
    mid_m_stride_b, mid_m_stride_h, mid_m_stride_s,
    state_stride_b, state_stride_h, state_stride_j, state_stride_d,
    stride_lg_b, stride_lg_h, stride_lg_n,
    stride_mg_b, stride_mg_h, stride_mg_n,
    stride_nw_h, stride_nw_d,
    stride_la_b, stride_la_h, stride_la_j,
    stride_y_b, stride_y_h, stride_y_n, stride_y_d,
    HAS_ALPHA: tl.constexpr,
    D_HEAD: tl.constexpr,
    J: tl.constexpr,
    J_PAD: tl.constexpr,
    SPLIT_K_MAX: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_d = tl.arange(0, D_HEAD)

    # Inlier GLA recurrence
    v_step = tl.load(V + batch_idx * stride_v_b + head_idx * stride_v_h + offs_d * stride_v_d).to(tl.float32)
    log_gamma = tl.load(Log_Gamma + batch_idx * stride_lg_b + head_idx * stride_lg_h).to(tl.float32)
    mix_g = tl.load(Mix_G + batch_idx * stride_mg_b + head_idx * stride_mg_h).to(tl.float32)
    gamma = tl.math.exp(log_gamma)

    offs_d_2d = tl.arange(0, D_HEAD)[None, :]
    y_i = tl.zeros([D_HEAD], dtype=tl.float32)
    BLOCK_J: tl.constexpr = 16
    for start_j in range(0, J, BLOCK_J):
        offs_j_chunk = start_j + tl.arange(0, BLOCK_J)
        mask_j_chunk = offs_j_chunk < J
        
        state_ptrs = State_I + batch_idx * state_stride_b + head_idx * state_stride_h + offs_j_chunk[:, None] * state_stride_j + offs_d_2d * state_stride_d
        state_chunk = tl.load(state_ptrs, mask=mask_j_chunk[:, None], other=0.0).to(tl.float32)
        
        k_chunk = tl.load(K + batch_idx * stride_k_b + head_idx * stride_k_h + (R + offs_j_chunk) * stride_k_d, mask=mask_j_chunk, other=0.0).to(tl.float32)
        q_chunk = tl.load(Q + batch_idx * stride_q_b + head_idx * stride_q_h + (R + offs_j_chunk) * stride_q_d, mask=mask_j_chunk, other=0.0).to(tl.float32)
        
        if HAS_ALPHA:
            la_ptrs = Log_Alpha + batch_idx * stride_la_b + head_idx * stride_la_h + offs_j_chunk * stride_la_j
            alpha_chunk = tl.load(la_ptrs, mask=mask_j_chunk, other=0.0).to(tl.float32)
            gamma_chunk = tl.math.exp(log_gamma * tl.math.exp(alpha_chunk))[:, None]
        else:
            gamma_chunk = gamma

        state_new_chunk = gamma_chunk * state_chunk + k_chunk[:, None] * v_step[None, :]
        y_i += tl.sum(q_chunk[:, None] * state_new_chunk, axis=0)
        
        state_new_ptrs = State_I_new + batch_idx * state_stride_b + head_idx * state_stride_h + offs_j_chunk[:, None] * state_stride_j + offs_d_2d * state_stride_d
        tl.store(state_new_ptrs, state_new_chunk, mask=mask_j_chunk[:, None])

    # Outlier Attention Reduction
    offs_s = tl.arange(0, SPLIT_K_MAX)
    mask_s = offs_s < split_k

    mid_m_base = Mid_M + batch_idx * mid_m_stride_b + head_idx * mid_m_stride_h
    mid_l_base = Mid_L + batch_idx * mid_l_stride_b + head_idx * mid_l_stride_h
    mid_o_base = Mid_O + batch_idx * mid_o_stride_b + head_idx * mid_o_stride_h

    m_s = tl.load(mid_m_base + offs_s * mid_m_stride_s, mask=mask_s, other=-float('inf'))
    global_m = tl.max(m_s, axis=0)

    diff = tl.where(mask_s & (m_s > -float('inf')), m_s - global_m, -float('inf'))
    alpha_s = tl.where(mask_s & (m_s > -float('inf')), tl.math.exp(diff), 0.0)

    l_s = tl.load(mid_l_base + offs_s * mid_l_stride_s, mask=mask_s, other=0.0)
    global_l = tl.sum(l_s * alpha_s, axis=0)

    offs_s_2d = tl.arange(0, SPLIT_K_MAX)[:, None]
    mask_2d = offs_s_2d < split_k
    o_all = tl.load(mid_o_base + offs_s_2d * mid_o_stride_s + offs_d_2d * mid_o_stride_d, mask=mask_2d, other=0.0)
    scale_2d = alpha_s[:, None]
    global_acc = tl.sum(o_all * scale_2d, axis=0)

    safe_l = tl.where(global_l > 0.0, global_l, 1.0)
    y_o = global_acc / safe_l

    # RMSNorm on inlier output
    var = tl.sum(y_i * y_i, axis=0) / D_HEAD
    rsqrt = tl.math.rsqrt(var + eps)
    norm_w = tl.load(Norm_W + head_idx * stride_nw_h + offs_d * stride_nw_d)
    y_i_norm = y_i * rsqrt * norm_w

    # Fuse outlier and inlier outputs
    y_out = mix_g * y_o + (1.0 - mix_g) * y_i_norm
    tl.store(Y + batch_idx * stride_y_b + head_idx * stride_y_h + offs_d * stride_y_d, y_out.to(Y.dtype.element_ty))

def fused_hofa_decode(
    q, k, v, 
    cache_k, cache_v, 
    state_I_in, state_I_out,
    log_gamma, mix_g,
    norm_w,
    R, seq_len, sm_scale,
    split_k=None,
    log_alpha=None
):
    B, H, N, D = q.shape
    assert N == 1

    total_heads = B * H

    HAS_ALPHA = log_alpha is not None
    if HAS_ALPHA:
        if log_alpha.ndim == 4:
            stride_la_b = log_alpha.stride(0) if log_alpha.shape[0] > 1 else 0
            stride_la_h = log_alpha.stride(1)
            stride_la_j = log_alpha.stride(3)
        elif log_alpha.ndim == 2:
            stride_la_b = 0
            stride_la_h = log_alpha.stride(0)
            stride_la_j = log_alpha.stride(1)
        else:
            stride_la_b, stride_la_h, stride_la_j = 0, log_alpha.shape[-1], 1
        log_alpha_tensor = log_alpha
    else:
        log_alpha_tensor = q
        stride_la_b, stride_la_h, stride_la_j = 0, 0, 0

    # Flash-Decoding split selection based on SM saturation and minimum work granularity
    if split_k is None:
        if R == 0:
            split_k = 1
        else:
            num_sms = _get_cached_sm_count(q.device)
            min_tokens_per_split = 1024
            max_splits = max(1, seq_len // min_tokens_per_split)
            target_splits = triton.cdiv(num_sms, total_heads)
            split_k = triton_next_power_of_2(min(target_splits, max_splits, 32))

    Y = torch.empty_like(q)
    eps = torch.finfo(torch.float32).eps if q.dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.finfo(q.dtype).eps

    BLOCK_HEADS = 1
    D_HEAD = triton.next_power_of_2(D)
    R_PAD = triton.next_power_of_2(R)
    J = D - R
    J_PAD = triton.next_power_of_2(J)

    if split_k <= 1:
        seq_len_bucket = get_bucket(seq_len)
        grid = (B, H)
        _fused_hofa_decode_kernel[grid](
            q, k, v,
            cache_k, cache_v,
            state_I_in,
            log_gamma, mix_g,
            norm_w,
            log_alpha_tensor,
            Y, state_I_out,
            R, seq_len, seq_len_bucket, sm_scale, eps,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            cache_k.stride(0), cache_k.stride(1), cache_k.stride(2), cache_k.stride(3),
            cache_v.stride(0), cache_v.stride(1), cache_v.stride(2), cache_v.stride(3),
            state_I_in.stride(0), state_I_in.stride(1), state_I_in.stride(2), state_I_in.stride(3),
            log_gamma.stride(0), log_gamma.stride(1), log_gamma.stride(2),
            mix_g.stride(0), mix_g.stride(1), mix_g.stride(2),
            norm_w.stride(0), norm_w.stride(1),
            stride_la_b, stride_la_h, stride_la_j,
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            HAS_ALPHA=HAS_ALPHA,
            BLOCK_HEADS=BLOCK_HEADS,
            D_HEAD=D_HEAD,
            R_PAD=R_PAD,
            J=J,
            J_PAD=J_PAD
        )
    else:
        tokens_per_split = triton.cdiv(seq_len, split_k)
        SPLIT_K_MAX = triton_next_power_of_2(split_k)
        tokens_per_split_bucket = get_bucket(tokens_per_split)

        mid_o, mid_l, mid_m = _get_cached_split_workspace(q.device, B, H, split_k, D_HEAD)

        grid_split = (B, H, split_k)
        _fused_hofa_decode_split_kernel[grid_split](
            q, cache_k, cache_v,
            mid_o, mid_l, mid_m,
            R, seq_len, tokens_per_split, tokens_per_split_bucket, sm_scale,
            split_k,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            cache_k.stride(0), cache_k.stride(1), cache_k.stride(2), cache_k.stride(3),
            cache_v.stride(0), cache_v.stride(1), cache_v.stride(2), cache_v.stride(3),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
            mid_l.stride(0), mid_l.stride(1), mid_l.stride(2),
            mid_m.stride(0), mid_m.stride(1), mid_m.stride(2),
            D_HEAD=D_HEAD,
            R_PAD=R_PAD,
        )

        grid_reduce = (B, H)
        _fused_hofa_decode_reduce_kernel[grid_reduce](
            q, k, v,
            mid_o, mid_l, mid_m,
            state_I_in, state_I_out,
            log_gamma, mix_g,
            norm_w,
            log_alpha_tensor,
            Y,
            R, split_k, eps,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
            mid_l.stride(0), mid_l.stride(1), mid_l.stride(2),
            mid_m.stride(0), mid_m.stride(1), mid_m.stride(2),
            state_I_in.stride(0), state_I_in.stride(1), state_I_in.stride(2), state_I_in.stride(3),
            log_gamma.stride(0), log_gamma.stride(1), log_gamma.stride(2),
            mix_g.stride(0), mix_g.stride(1), mix_g.stride(2),
            norm_w.stride(0), norm_w.stride(1),
            stride_la_b, stride_la_h, stride_la_j,
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            HAS_ALPHA=HAS_ALPHA,
            D_HEAD=D_HEAD,
            J=J,
            J_PAD=J_PAD,
            SPLIT_K_MAX=SPLIT_K_MAX,
        )

    return Y
