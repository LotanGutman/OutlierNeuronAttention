import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from src.modules.triton_utils import get_device_max_sram, get_hofa_bwd_chunk_size

def hofa_bwd_early_prune(configs, named_args, **kwargs):
    device = named_args['Q_ptr'].device
    max_sram = get_device_max_sram(device)
    
    j_padded = named_args.get('j_padded', kwargs.get('j_padded'))
    d_head_padded = named_args.get('d_head_padded', kwargs.get('d_head_padded'))
    chunk_size = named_args.get('CHUNK_SIZE', kwargs.get('CHUNK_SIZE'))
    
    state_bytes = j_padded * d_head_padded * 4
    
    pruned_configs = []
    for config in configs:
        block_m = config.kwargs['BLOCK_M']
        if block_m != chunk_size:
            continue
            
        chunk_bytes = 8 * block_m**2 + (4 * j_padded + 10 * d_head_padded) * block_m
        if state_bytes + chunk_bytes <= max_sram:
            pruned_configs.append(config)
            
    return pruned_configs

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['N', 'r', 'j', 'd_head'],
    prune_configs_by={
        'early_config_prune': hofa_bwd_early_prune,
        'perf_model': None,
        'top_k': None
    }
)
@triton.jit
def hofa_bwd_router_kernel(
    Q_ptr, K_ptr, V_ptr, log_gamma_ptr, states_in_ptr,
    dY_out_ptr, mix_g_ptr, gla_scale_ptr,
    dQ_ptr,
    d_mix_g_ptr, dy_ptr, D_ptr, LSE_ptr,
    N, r, j, j_padded: tl.constexpr, d_head, d_head_padded: tl.constexpr,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_log_gamma_b, stride_log_gamma_h, stride_log_gamma_n,
    stride_state_u, stride_state_in_b, stride_state_in_h, stride_state_in_j, stride_state_in_d,
    stride_dy_out_b, stride_dy_out_h, stride_dy_out_n, stride_dy_out_d,
    stride_mix_g_b, stride_mix_g_h, stride_mix_g_n,
    stride_gla_scale_h, stride_gla_scale_d,
    stride_dq_b, stride_dq_h, stride_dq_n, stride_dq_d,
    stride_dmix_g_b, stride_dmix_g_h, stride_dmix_g_n,
    stride_dy_b, stride_dy_h, stride_dy_n, stride_dy_d,
    stride_D_b, stride_D_h, stride_D_n,
    stride_LSE_b, stride_LSE_h, stride_LSE_n,
    eps, sm_scale,
    H,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr
):
    pid_u = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    offsets_c = tl.arange(0, BLOCK_M)
    offsets_j = tl.arange(0, j_padded)
    offsets_d = tl.arange(0, BLOCK_DMODEL_V)
    
    mask_j = offsets_j < j
    mask_d = offsets_d < d_head
    
    inlier_indices = r + offsets_j
    
    start = pid_u * BLOCK_M
    t_offs = start + offsets_c
    mask_t = t_offs < N
    
    # Base pointers
    b_h_q_ptr = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    b_h_k_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    b_h_v_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h
    b_h_log_gamma_ptr = log_gamma_ptr + pid_b * stride_log_gamma_b + pid_h * stride_log_gamma_h
    b_h_state_in_ptr = states_in_ptr + pid_b * stride_state_in_b + pid_h * stride_state_in_h
    
    b_h_dy_out_ptr = dY_out_ptr + pid_b * stride_dy_out_b + pid_h * stride_dy_out_h
    b_h_mix_g_ptr = mix_g_ptr + pid_b * stride_mix_g_b + pid_h * stride_mix_g_h
    b_h_gla_scale_ptr = gla_scale_ptr + pid_h * stride_gla_scale_h
    
    b_h_dmix_g_ptr = d_mix_g_ptr + pid_b * stride_dmix_g_b + pid_h * stride_dmix_g_h
    b_h_dy_ptr = dy_ptr + pid_b * stride_dy_b + pid_h * stride_dy_h
    b_h_D_ptr = D_ptr + pid_b * stride_D_b + pid_h * stride_D_h
    b_h_LSE_ptr = LSE_ptr + pid_b * stride_LSE_b + pid_h * stride_LSE_h
    
    # 1. Recompute raw inlier pathway output Y_GLA in SRAM
    q_ptrs = b_h_q_ptr + t_offs[:, None] * stride_q_n + inlier_indices[None, :] * stride_q_d
    Q_J = tl.load(q_ptrs, mask=mask_t[:, None] & mask_j[None, :], other=0.0)
    
    k_ptrs = b_h_k_ptr + t_offs[:, None] * stride_k_n + inlier_indices[None, :] * stride_k_d
    K_J = tl.load(k_ptrs, mask=mask_t[:, None] & mask_j[None, :], other=0.0)
    
    v_ptrs = b_h_v_ptr + t_offs[:, None] * stride_v_n + offsets_d[None, :] * stride_v_d
    V_c = tl.load(v_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0)
    
    log_gamma_ptrs = b_h_log_gamma_ptr + t_offs * stride_log_gamma_n
    g_c = tl.load(log_gamma_ptrs, mask=mask_t, other=0.0).to(tl.float32)
    
    state_in_ptrs = b_h_state_in_ptr + pid_u * stride_state_u + offsets_j[:, None] * stride_state_in_j + offsets_d[None, :] * stride_state_in_d
    S_in = tl.load(state_in_ptrs, mask=mask_j[:, None] & mask_d[None, :], other=0.0)
    
    g_cumsum = tl.cumsum(g_c, axis=0)
    diff = g_cumsum[:, None] - g_cumsum[None, :]
    diff = tl.where(offsets_c[:, None] >= offsets_c[None, :], diff, -float('inf'))
    decay_mask = tl.math.exp(diff)
    
    attn = tl.dot(Q_J.to(V_c.dtype), tl.trans(K_J.to(V_c.dtype)), allow_tf32=False)
    attn = (attn * decay_mask).to(V_c.dtype)
    Y_intra = tl.dot(attn, V_c, allow_tf32=False)
    
    Y_inter = tl.dot(Q_J.to(V_c.dtype), S_in.to(V_c.dtype), allow_tf32=False) * tl.math.exp(g_cumsum)[:, None]
    Y_GLA = Y_intra + Y_inter
    
    Y_GLA_masked = tl.where(mask_d[None, :], Y_GLA, 0.0)
    rms = tl.sqrt(tl.sum(Y_GLA_masked * Y_GLA_masked, axis=1) / d_head + eps)
    Y_GLA_norm = Y_GLA_masked / rms[:, None]
    
    gla_scale_ptrs = b_h_gla_scale_ptr + offsets_d * stride_gla_scale_d
    w = tl.load(gla_scale_ptrs, mask=mask_d, other=0.0)
    Y_I = Y_GLA_norm * w[None, :]
    
    # 2. Recompute exact attention output Y_O in SRAM
    offsets_r = tl.arange(0, BLOCK_DMODEL_QK)
    mask_r = offsets_r < r
    q_o_ptrs = b_h_q_ptr + t_offs[:, None] * stride_q_n + offsets_r[None, :] * stride_q_d
    Q_O = tl.load(q_o_ptrs, mask=mask_t[:, None] & mask_r[None, :], other=0.0)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=tl.float32)
    
    for w_start in range(0, start + BLOCK_M, BLOCK_N):
        offs_n = w_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        k_o_ptrs = b_h_k_ptr + offs_n[None, :] * stride_k_n + offsets_r[:, None] * stride_k_d
        K_O_w = tl.load(k_o_ptrs, mask=mask_n[None, :] & mask_r[:, None], other=0.0)
        
        v_ptrs_w = b_h_v_ptr + offs_n[:, None] * stride_v_n + offsets_d[None, :] * stride_v_d
        V_w = tl.load(v_ptrs_w, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        S = tl.dot(Q_O.to(K_O_w.dtype), K_O_w, allow_tf32=False) * sm_scale
        # Correctly mask causally using absolute sequence positions
        S = tl.where(t_offs[:, None] >= offs_n[None, :], S, float('-inf'))
            
        m_ij = tl.maximum(m_i, tl.max(S, 1))
        p = tl.exp(S - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + tl.dot(p.to(V_w.dtype), V_w, allow_tf32=False)
        m_i = m_ij
        
    Y_O = acc / l_i[:, None]
    LSE = m_i + tl.log(l_i)
    
    # 3. Compute gradients in SRAM
    dy_out_ptrs = b_h_dy_out_ptr + t_offs[:, None] * stride_dy_out_n + offsets_d[None, :] * stride_dy_out_d
    dY_out = tl.load(dy_out_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0)
    
    mix_g_ptrs = b_h_mix_g_ptr + t_offs[:, None] * stride_mix_g_n
    mix_g = tl.load(mix_g_ptrs, mask=mask_t[:, None], other=0.0)
    
    d_mix_g_val = tl.sum(dY_out * (Y_O - Y_I), axis=1)
    
    dY_O = dY_out * mix_g
    dY_I = dY_out * (1.0 - mix_g)
    
    D_val = tl.sum(dY_O * Y_O, axis=1)
    
    dy_norm = dY_I * w[None, :]
    sum_dy_norm_y = tl.sum(dy_norm * Y_GLA_norm, axis=1)
    dy = (dy_norm - Y_GLA_norm * (sum_dy_norm_y[:, None] / d_head)) / rms[:, None]
    dy = tl.where(mask_d[None, :], dy, 0.0)
    
    # 3.5. Compute exact attention query gradients dQ_O
    dQ_O_acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_QK], dtype=tl.float32)
    for w_start in range(0, start + BLOCK_M, BLOCK_N):
        offs_n = w_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        k_o_ptrs = b_h_k_ptr + offs_n[None, :] * stride_k_n + offsets_r[:, None] * stride_k_d
        K_O_w = tl.load(k_o_ptrs, mask=mask_n[None, :] & mask_r[:, None], other=0.0)
        
        v_ptrs_w = b_h_v_ptr + offs_n[:, None] * stride_v_n + offsets_d[None, :] * stride_v_d
        V_w = tl.load(v_ptrs_w, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        S = tl.dot(Q_O.to(K_O_w.dtype), K_O_w, allow_tf32=False) * sm_scale
        S = tl.where(t_offs[:, None] >= offs_n[None, :], S, float('-inf'))
        P = tl.exp(S - LSE[:, None])
        
        dY_O_V = tl.dot(dY_O.to(V_w.dtype), tl.trans(V_w), allow_tf32=False)
        dP = P * (dY_O_V - D_val[:, None])
        
        dQ_O_acc += tl.dot(dP.to(K_O_w.dtype), tl.trans(K_O_w), allow_tf32=False) * sm_scale
        
    # 4. Write outputs to HBM
    dmix_g_ptrs = b_h_dmix_g_ptr + t_offs[:, None] * stride_dmix_g_n
    tl.store(dmix_g_ptrs, d_mix_g_val[:, None], mask=mask_t[:, None])
    
    dy_ptrs = b_h_dy_ptr + t_offs[:, None] * stride_dy_n + offsets_d[None, :] * stride_dy_d
    tl.store(dy_ptrs, dy, mask=mask_t[:, None] & mask_d[None, :])
    
    D_ptrs = b_h_D_ptr + t_offs * stride_D_n
    tl.store(D_ptrs, D_val, mask=mask_t)
    
    LSE_ptrs = b_h_LSE_ptr + t_offs * stride_LSE_n
    tl.store(LSE_ptrs, LSE, mask=mask_t)
    
    b_h_dQ_ptr = dQ_ptr + pid_b * stride_dq_b + pid_h * stride_dq_h
    dq_o_ptrs = b_h_dQ_ptr + t_offs[:, None] * stride_dq_n + offsets_r[None, :] * stride_dq_d
    tl.store(dq_o_ptrs, dQ_O_acc.to(dQ_ptr.dtype.element_ty), mask=mask_t[:, None] & mask_r[None, :])
    
    # 5. Reconstruct states and compute inlier Q gradient (dQ_J) token-by-token
    S_curr = S_in
    for i in range(BLOCK_M):
        mask_ti = start + i < N
        if mask_ti:
            log_gamma_i = tl.sum(tl.where(tl.arange(0, BLOCK_M) == i, g_c, 0.0))
            gamma_i = tl.exp(log_gamma_i)
            
            K_i_ptrs = b_h_k_ptr + (start + i) * stride_k_n + inlier_indices * stride_k_d
            V_i_ptrs = b_h_v_ptr + (start + i) * stride_v_n + offsets_d * stride_v_d
            K_i = tl.load(K_i_ptrs, mask=mask_j, other=0.0).to(tl.float32)
            V_i = tl.load(V_i_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            
            S_curr = S_curr * gamma_i + K_i[:, None] * V_i[None, :]
            
            dy_i = tl.load(b_h_dy_ptr + (start + i) * stride_dy_n + offsets_d * stride_dy_d, mask=mask_d, other=0.0).to(tl.float32)
            
            dQ_J_i = tl.sum(dy_i[None, :] * S_curr, axis=1)
            
            dq_j_ptr_i = b_h_dQ_ptr + (start + i) * stride_dq_n + inlier_indices * stride_dq_d
            tl.store(dq_j_ptr_i, dQ_J_i.to(dQ_ptr.dtype.element_ty), mask=mask_j)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['N', 'r', 'j', 'd_head'],
    prune_configs_by={
        'early_config_prune': hofa_bwd_early_prune,
        'perf_model': None,
        'top_k': None
    },
    restore_value=[]
)
@triton.jit
def hofa_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, log_gamma_ptr, states_in_ptr,
    dy_ptr, dY_out_ptr, mix_g_ptr, D_ptr, LSE_ptr,
    dK_ptr, dV_ptr, dlog_gamma_ptr,
    N, r, j, j_padded: tl.constexpr, d_head, d_head_padded: tl.constexpr,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_log_gamma_b, stride_log_gamma_h, stride_log_gamma_n,
    stride_state_u, stride_state_in_b, stride_state_in_h, stride_state_in_j, stride_state_in_d,
    stride_dy_b, stride_dy_h, stride_dy_n, stride_dy_d,
    stride_dy_out_b, stride_dy_out_h, stride_dy_out_n, stride_dy_out_d,
    stride_mix_g_b, stride_mix_g_h, stride_mix_g_n,
    stride_D_b, stride_D_h, stride_D_n,
    stride_LSE_b, stride_LSE_h, stride_LSE_n,
    stride_dk_b, stride_dk_h, stride_dk_n, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_n, stride_dv_d,
    stride_dlog_gamma_b, stride_dlog_gamma_h, stride_dlog_gamma_n,
    sm_scale,
    H,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr
):
    pid_v = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    offsets_c = tl.arange(0, BLOCK_M)
    offsets_j = tl.arange(0, j_padded)
    offsets_d = tl.arange(0, BLOCK_DMODEL_V)
    
    mask_j = offsets_j < j
    mask_d = offsets_d < d_head
    inlier_indices = r + offsets_j
    
    start_v = pid_v * BLOCK_M
    t_offs_v = start_v + offsets_c
    mask_tv = t_offs_v < N
    
    # Base pointers
    b_h_q_ptr = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    b_h_k_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    b_h_v_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h
    b_h_log_gamma_ptr = log_gamma_ptr + pid_b * stride_log_gamma_b + pid_h * stride_log_gamma_h
    b_h_state_in_ptr = states_in_ptr + pid_b * stride_state_in_b + pid_h * stride_state_in_h
    
    b_h_dy_ptr = dy_ptr + pid_b * stride_dy_b + pid_h * stride_dy_h
    b_h_dy_out_ptr = dY_out_ptr + pid_b * stride_dy_out_b + pid_h * stride_dy_out_h
    b_h_mix_g_ptr = mix_g_ptr + pid_b * stride_mix_g_b + pid_h * stride_mix_g_h
    b_h_D_ptr = D_ptr + pid_b * stride_D_b + pid_h * stride_D_h
    b_h_LSE_ptr = LSE_ptr + pid_b * stride_LSE_b + pid_h * stride_LSE_h
    
    b_h_dK_ptr = dK_ptr + pid_b * stride_dk_b + pid_h * stride_dk_h
    b_h_dV_ptr = dV_ptr + pid_b * stride_dv_b + pid_h * stride_dv_h
    b_h_dlog_gamma_ptr = dlog_gamma_ptr + pid_b * stride_dlog_gamma_b + pid_h * stride_dlog_gamma_h
    
    # Load Key block inputs
    k_j_ptrs = b_h_k_ptr + t_offs_v[:, None] * stride_k_n + inlier_indices[None, :] * stride_k_d
    K_J_v = tl.load(k_j_ptrs, mask=mask_tv[:, None] & mask_j[None, :], other=0.0)
    
    offsets_r = tl.arange(0, BLOCK_DMODEL_QK)
    mask_r = offsets_r < r
    k_o_ptrs = b_h_k_ptr + t_offs_v[:, None] * stride_k_n + offsets_r[None, :] * stride_k_d
    K_O_v = tl.load(k_o_ptrs, mask=mask_tv[:, None] & mask_r[None, :], other=0.0)
    
    v_ptrs = b_h_v_ptr + t_offs_v[:, None] * stride_v_n + offsets_d[None, :] * stride_v_d
    V_v = tl.load(v_ptrs, mask=mask_tv[:, None] & mask_d[None, :], other=0.0)
    
    log_gamma_ptrs = b_h_log_gamma_ptr + t_offs_v * stride_log_gamma_n
    g_v = tl.load(log_gamma_ptrs, mask=mask_tv, other=0.0).to(tl.float32)
    
    state_in_ptrs = b_h_state_in_ptr + pid_v * stride_state_u + offsets_j[:, None] * stride_state_in_j + offsets_d[None, :] * stride_state_in_d
    S_in_v = tl.load(state_in_ptrs, mask=mask_j[:, None] & mask_d[None, :], other=0.0)
    
    # Initialize accumulators
    dK_O_acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_QK], dtype=tl.float32)
    dK_J_acc = tl.zeros([BLOCK_M, j_padded], dtype=tl.float32)
    dV_acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=tl.float32)
    dlog_gamma_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    dS = tl.zeros([j_padded, BLOCK_DMODEL_V], dtype=tl.float32)
    
    # Backward loop over future blocks w from U-1 down to v+1
    U = tl.cdiv(N, BLOCK_M)
    for w in range(U - 1, pid_v, -1):
        start_w = w * BLOCK_M
        t_offs_w = start_w + offsets_c
        mask_tw = t_offs_w < N
        
        q_j_ptrs = b_h_q_ptr + t_offs_w[:, None] * stride_q_n + inlier_indices[None, :] * stride_q_d
        Q_J_w = tl.load(q_j_ptrs, mask=mask_tw[:, None] & mask_j[None, :], other=0.0)
        
        dy_ptrs = b_h_dy_ptr + t_offs_w[:, None] * stride_dy_n + offsets_d[None, :] * stride_dy_d
        dy_w = tl.load(dy_ptrs, mask=mask_tw[:, None] & mask_d[None, :], other=0.0)
        
        log_gamma_ptrs_w = b_h_log_gamma_ptr + t_offs_w * stride_log_gamma_n
        g_w = tl.load(log_gamma_ptrs_w, mask=mask_tw, other=0.0).to(tl.float32)
        
        g_cumsum_w = tl.cumsum(g_w, axis=0)
        g_last_w = tl.sum(g_w)
        
        q_decay = Q_J_w * tl.exp(g_cumsum_w)[:, None]
        q_decay = tl.where(mask_tw[:, None], q_decay, 0.0)
        
        dS = dS * tl.exp(g_last_w) + tl.dot(tl.trans(q_decay.to(dy_w.dtype)), dy_w, allow_tf32=False)
        
        # Exact attention pathway for w > v
        q_o_ptrs = b_h_q_ptr + t_offs_w[:, None] * stride_q_n + offsets_r[None, :] * stride_q_d
        Q_O_w = tl.load(q_o_ptrs, mask=mask_tw[:, None] & mask_r[None, :], other=0.0)
        
        dy_out_ptrs = b_h_dy_out_ptr + t_offs_w[:, None] * stride_dy_out_n + offsets_d[None, :] * stride_dy_out_d
        dY_out_w = tl.load(dy_out_ptrs, mask=mask_tw[:, None] & mask_d[None, :], other=0.0)
        
        mix_g_ptrs_w = b_h_mix_g_ptr + t_offs_w[:, None] * stride_mix_g_n
        mix_g_w = tl.load(mix_g_ptrs_w, mask=mask_tw[:, None], other=0.0)
        
        dY_O_w = dY_out_w * mix_g_w
        
        LSE_ptrs_w = b_h_LSE_ptr + t_offs_w * stride_LSE_n
        LSE_w = tl.load(LSE_ptrs_w, mask=mask_tw, other=0.0)
        
        D_ptrs_w = b_h_D_ptr + t_offs_w * stride_D_n
        D_w = tl.load(D_ptrs_w, mask=mask_tw, other=0.0)
        
        S_O = tl.dot(Q_O_w.to(K_O_v.dtype), tl.trans(K_O_v), allow_tf32=False) * sm_scale
        P_O = tl.exp(S_O - LSE_w[:, None])
        P_O = tl.where(mask_tw[:, None] & mask_tv[None, :], P_O, 0.0)
        
        dV_acc += tl.dot(tl.trans(P_O.to(dY_O_w.dtype)), dY_O_w, allow_tf32=False)
        
        dY_O_V = tl.dot(dY_O_w.to(V_v.dtype), tl.trans(V_v), allow_tf32=False)
        dP_O = P_O * (dY_O_V - D_w[:, None])
        
        dK_O_acc += tl.dot(tl.trans(dP_O.to(Q_O_w.dtype)), Q_O_w, allow_tf32=False) * sm_scale
        
    # Process local block w == v
    q_o_ptrs_v = b_h_q_ptr + t_offs_v[:, None] * stride_q_n + offsets_r[None, :] * stride_q_d
    Q_O_v = tl.load(q_o_ptrs_v, mask=mask_tv[:, None] & mask_r[None, :], other=0.0)
    
    dy_out_ptrs_v = b_h_dy_out_ptr + t_offs_v[:, None] * stride_dy_out_n + offsets_d[None, :] * stride_dy_out_d
    dY_out_v = tl.load(dy_out_ptrs_v, mask=mask_tv[:, None] & mask_d[None, :], other=0.0)
    
    mix_g_ptrs_v = b_h_mix_g_ptr + t_offs_v[:, None] * stride_mix_g_n
    mix_g_v = tl.load(mix_g_ptrs_v, mask=mask_tv[:, None], other=0.0)
    dY_O_v = dY_out_v * mix_g_v
    
    LSE_v = tl.load(b_h_LSE_ptr + t_offs_v * stride_LSE_n, mask=mask_tv, other=0.0)
    D_v = tl.load(b_h_D_ptr + t_offs_v * stride_D_n, mask=mask_tv, other=0.0)
    
    S_O_v = tl.dot(Q_O_v.to(K_O_v.dtype), tl.trans(K_O_v), allow_tf32=False) * sm_scale
    S_O_v = tl.where(offsets_c[:, None] >= offsets_c[None, :], S_O_v, float('-inf'))
    
    P_O_v = tl.exp(S_O_v - LSE_v[:, None])
    P_O_v = tl.where(mask_tv[:, None] & mask_tv[None, :], P_O_v, 0.0)
    
    dV_acc += tl.dot(tl.trans(P_O_v.to(dY_O_v.dtype)), dY_O_v, allow_tf32=False)
    
    dY_O_V_v = tl.dot(dY_O_v.to(V_v.dtype), tl.trans(V_v), allow_tf32=False)
    dP_O_v = P_O_v * (dY_O_V_v - D_v[:, None])
    
    dK_O_acc += tl.dot(tl.trans(dP_O_v.to(Q_O_v.dtype)), Q_O_v, allow_tf32=False) * sm_scale
    
    # Process local GLA scan inside chunk v
    dy_ptrs_v = b_h_dy_ptr + t_offs_v[:, None] * stride_dy_n + offsets_d[None, :] * stride_dy_d
    dy_v = tl.load(dy_ptrs_v, mask=mask_tv[:, None] & mask_d[None, :], other=0.0)
    
    q_j_ptrs_v = b_h_q_ptr + t_offs_v[:, None] * stride_q_n + inlier_indices[None, :] * stride_q_d
    Q_J_v = tl.load(q_j_ptrs_v, mask=mask_tv[:, None] & mask_j[None, :], other=0.0)
    
    for i in range(BLOCK_M - 1, -1, -1):
        mask_ti = start_v + i < N
        if mask_ti:
            S_prev = S_in_v
            for s in range(0, i):
                log_gamma_s = tl.sum(tl.where(tl.arange(0, BLOCK_M) == s, g_v, 0.0))
                gamma_s = tl.exp(log_gamma_s)
                K_s_ptrs = b_h_k_ptr + (start_v + s) * stride_k_n + inlier_indices * stride_k_d
                V_s_ptrs = b_h_v_ptr + (start_v + s) * stride_v_n + offsets_d * stride_v_d
                K_s = tl.load(K_s_ptrs, mask=mask_j, other=0.0).to(tl.float32)
                V_s = tl.load(V_s_ptrs, mask=mask_d, other=0.0).to(tl.float32)
                S_prev = S_prev * gamma_s + K_s[:, None] * V_s[None, :]
                
            log_gamma_i = tl.sum(tl.where(tl.arange(0, BLOCK_M) == i, g_v, 0.0))
            gamma_i = tl.exp(log_gamma_i)
            
            K_i_ptrs = b_h_k_ptr + (start_v + i) * stride_k_n + inlier_indices * stride_k_d
            V_i_ptrs = b_h_v_ptr + (start_v + i) * stride_v_n + offsets_d * stride_v_d
            K_i = tl.load(K_i_ptrs, mask=mask_j, other=0.0).to(tl.float32)
            V_i = tl.load(V_i_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            
            dy_i = tl.load(b_h_dy_ptr + (start_v + i) * stride_dy_n + offsets_d * stride_dy_d, mask=mask_d, other=0.0).to(tl.float32)
            Q_i = tl.load(b_h_q_ptr + (start_v + i) * stride_q_n + inlier_indices * stride_q_d, mask=mask_j, other=0.0).to(tl.float32)
            
            # 1. Add local query gradient contribution
            dS += Q_i[:, None] * dy_i[None, :]
            
            # 2. Compute gradients for token i
            dK_J_i = tl.sum(V_i[None, :] * dS, axis=1)
            dV_i = tl.sum(K_i[:, None] * dS, axis=0)
            dlog_gamma_i = gamma_i * tl.sum(dS * S_prev)
            
            dK_J_acc = tl.where(tl.arange(0, BLOCK_M)[:, None] == i, dK_J_i[None, :], dK_J_acc)
            dV_acc += tl.where(tl.arange(0, BLOCK_M)[:, None] == i, dV_i[None, :], 0.0)
            dlog_gamma_acc = tl.where(tl.arange(0, BLOCK_M) == i, dlog_gamma_i, dlog_gamma_acc)
            
            # 3. Propagate to previous token
            dS = dS * gamma_i
            
    # Write Key block outputs to HBM
    dk_o_ptrs = b_h_dK_ptr + t_offs_v[:, None] * stride_dk_n + offsets_r[None, :] * stride_dk_d
    tl.store(dk_o_ptrs, dK_O_acc.to(dK_ptr.dtype.element_ty), mask=mask_tv[:, None] & mask_r[None, :])
    
    dk_j_ptrs = b_h_dK_ptr + t_offs_v[:, None] * stride_dk_n + inlier_indices[None, :] * stride_dk_d
    tl.store(dk_j_ptrs, dK_J_acc.to(dK_ptr.dtype.element_ty), mask=mask_tv[:, None] & mask_j[None, :])
    
    dv_ptrs = b_h_dV_ptr + t_offs_v[:, None] * stride_dv_n + offsets_d[None, :] * stride_dv_d
    tl.store(dv_ptrs, dV_acc.to(dV_ptr.dtype.element_ty), mask=mask_tv[:, None] & mask_d[None, :])
    
    dlog_gamma_ptrs = b_h_dlog_gamma_ptr + t_offs_v * stride_dlog_gamma_n
    tl.store(dlog_gamma_ptrs, dlog_gamma_acc.to(dlog_gamma_ptr.dtype.element_ty), mask=mask_tv)


class HOFAAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, gate_logits, mix_g, gla_scale, r, chunk_size):
        B, H, N, d_head = Q.shape
        j = d_head - r
        device = Q.device
        
        j_padded = 1 << (j - 1).bit_length()
        j_padded = max(j_padded, 16)
        d_head_padded = 1 << (d_head - 1).bit_length()
        d_head_padded = max(d_head_padded, 16)
        
        # Query device max shared memory using helper utility
        max_sram = get_device_max_sram(device)
        # Compute safe chunk size using helper utility
        chunk_size = get_hofa_bwd_chunk_size(j_padded, d_head_padded, max_sram, initial_chunk_size=chunk_size)
        
        log_gamma = F.logsigmoid(-gate_logits).squeeze(-1)
        
        U = (N + chunk_size - 1) // chunk_size
        requires_grad = Q.requires_grad or K.requires_grad or V.requires_grad or gate_logits.requires_grad or mix_g.requires_grad or gla_scale.requires_grad
        
        if requires_grad:
            states_in = torch.empty(B, H, U, j, d_head, device=device, dtype=torch.float32)
        else:
            states_in = torch.empty(1, 1, 1, 1, 1, device=device, dtype=torch.float32)
            
        Y_GLA = torch.zeros(B, H, N, d_head, dtype=Q.dtype, device=device)
        
        from src.chunk_gla_inlier import chunk_gla_fwd_kernel
        grid = (B, H)
        chunk_gla_fwd_kernel[grid](
            Q, K, V, log_gamma,
            states_in,
            Y_GLA,
            N, chunk_size,
            r, j, j_padded, d_head, d_head_padded,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            log_gamma.stride(0), log_gamma.stride(1), log_gamma.stride(2),
            states_in.stride(2), states_in.stride(0), states_in.stride(1), states_in.stride(3), states_in.stride(4),
            Y_GLA.stride(0), Y_GLA.stride(1), Y_GLA.stride(2), Y_GLA.stride(3),
            REQUIRES_GRAD=requires_grad
        )
        
        rms = torch.rsqrt(Y_GLA.pow(2).sum(dim=-1, keepdim=True) / d_head + 1e-5)
        Y_I_normed = Y_GLA * rms
        Y_I = Y_I_normed * gla_scale
        
        Y_Final = (1.0 - mix_g) * Y_I
        
        sm_scale = (d_head / r) ** 0.5
        from src.exact_attention import exact_attention_triton
        exact_attention_triton(Q, K, V, r, sm_scale, mix_g.squeeze(-1), out=Y_Final)
        
        if requires_grad:
            ctx.save_for_backward(Q, K, V, gate_logits, mix_g, gla_scale, states_in, Y_GLA)
            ctx.r = r
            ctx.chunk_size = chunk_size
            ctx.j = j
            ctx.j_padded = j_padded
            ctx.d_head = d_head
            ctx.d_head_padded = d_head_padded
            ctx.sm_scale = sm_scale
            
        return Y_Final

    @staticmethod
    def backward(ctx, dY_out):
        Q, K, V, gate_logits, mix_g, gla_scale, states_in, Y_GLA = ctx.saved_tensors
        r = ctx.r
        chunk_size = ctx.chunk_size
        j = ctx.j
        j_padded = ctx.j_padded
        d_head = ctx.d_head
        d_head_padded = ctx.d_head_padded
        sm_scale = ctx.sm_scale
        
        B, H, N, _ = Q.shape
        device = Q.device
        
        d_mix_g = torch.empty(B, H, N, 1, dtype=Q.dtype, device=device)
        dy = torch.empty(B, H, N, d_head, dtype=Q.dtype, device=device)
        D = torch.empty(B, H, N, dtype=torch.float32, device=device)
        LSE = torch.empty(B, H, N, dtype=torch.float32, device=device)
        dQ = torch.zeros_like(Q)
        
        grid_router = (triton.cdiv(N, chunk_size), B * H)
        log_gamma = F.logsigmoid(-gate_logits).squeeze(-1)
        
        hofa_bwd_router_kernel[grid_router](
            Q, K, V, log_gamma, states_in,
            dY_out, mix_g, gla_scale,
            dQ,
            d_mix_g, dy, D, LSE,
            N, r, j, j_padded, d_head, d_head_padded,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            log_gamma.stride(0), log_gamma.stride(1), log_gamma.stride(2),
            states_in.stride(2), states_in.stride(0), states_in.stride(1), states_in.stride(3), states_in.stride(4),
            dY_out.stride(0), dY_out.stride(1), dY_out.stride(2), dY_out.stride(3),
            mix_g.stride(0), mix_g.stride(1), mix_g.stride(2),
            gla_scale.stride(1), gla_scale.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            d_mix_g.stride(0), d_mix_g.stride(1), d_mix_g.stride(2),
            dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3),
            D.stride(0), D.stride(1), D.stride(2),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            1e-5, sm_scale,
            H=H,
            BLOCK_DMODEL_QK=max(triton.next_power_of_2(r), 16),
            BLOCK_DMODEL_V=d_head_padded,
            CHUNK_SIZE=chunk_size
        )
        
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        dlog_gamma = torch.empty(B, H, N, dtype=Q.dtype, device=device)
        
        grid_mono = (triton.cdiv(N, chunk_size), B * H)
        
        hofa_bwd_dkv_kernel[grid_mono](
            Q, K, V, log_gamma, states_in,
            dy, dY_out, mix_g, D, LSE,
            dK, dV, dlog_gamma,
            N, r, j, j_padded, d_head, d_head_padded,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            log_gamma.stride(0), log_gamma.stride(1), log_gamma.stride(2),
            states_in.stride(2), states_in.stride(0), states_in.stride(1), states_in.stride(3), states_in.stride(4),
            dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3),
            dY_out.stride(0), dY_out.stride(1), dY_out.stride(2), dY_out.stride(3),
            mix_g.stride(0), mix_g.stride(1), mix_g.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            dlog_gamma.stride(0), dlog_gamma.stride(1), dlog_gamma.stride(2),
            sm_scale,
            H=H,
            BLOCK_DMODEL_QK=max(triton.next_power_of_2(r), 16),
            BLOCK_DMODEL_V=d_head_padded,
            CHUNK_SIZE=chunk_size
        )
        
        rms = torch.rsqrt(Y_GLA.pow(2).sum(dim=-1, keepdim=True) / d_head + 1e-5)
        Y_I_normed = Y_GLA * rms
        dY_I = dY_out * (1.0 - mix_g)
        dgla_scale = (dY_I * Y_I_normed).sum(dim=(0, 2), keepdim=True)
        
        dgate_logits = (dlog_gamma * -(1.0 - torch.exp(log_gamma))).unsqueeze(-1)
        
        return dQ, dK, dV, dgate_logits, d_mix_g, dgla_scale, None, None
