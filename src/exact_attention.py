import torch
import triton
import triton.language as tl
from src.modules.triton_utils import get_device_max_sram, get_exact_attn_block_sizes, get_bucket, BUCKETS

def exact_attn_early_prune(configs, named_args, **kwargs):
    device = named_args['Q'].device
    max_sram = get_device_max_sram(device)
    
    block_dmodel_v = named_args.get('BLOCK_DMODEL_V', kwargs.get('BLOCK_DMODEL_V'))
    block_dmodel_qk = named_args.get('BLOCK_DMODEL_QK', kwargs.get('BLOCK_DMODEL_QK'))
    
    pruned_configs = []
    for config in configs:
        block_m = config.kwargs['BLOCK_M']
        block_n = config.kwargs['BLOCK_N']
        # Estimate peak SRAM: acc[M,V]*4 + q[M,QK]*2 + k[N,QK]*2 + v[N,V]*2 + qk[M,N]*4
        est = (block_m * block_dmodel_v * 4 + block_m * block_dmodel_qk * 2
               + block_n * block_dmodel_qk * 2 + block_n * block_dmodel_v * 2
               + block_m * block_n * 4)
        if est <= max_sram:
            pruned_configs.append(config)
    return pruned_configs

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=2, num_stages=4),
    ],
    key=['N_CTX_bucket', 'r'],
    prune_configs_by={
        'early_config_prune': exact_attn_early_prune,
        'perf_model': None,
        'top_k': None
    },
    restore_value=['Out']
)
@triton.jit
def _fwd_kernel(
    Q, K, V, Out, mix_g_ptr,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_mix_g_b, stride_mix_g_h, stride_mix_g_n,
    Z, H, N_CTX, r, N_CTX_bucket,
    Part_Out_ptr, Part_L_ptr, Part_M_ptr,
    stride_part_out_zh, stride_part_out_k, stride_part_out_n, stride_part_out_d,
    stride_part_l_zh, stride_part_l_k, stride_part_l_n,
    stride_part_m_zh, stride_part_m_k, stride_part_m_n,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL_QK: tl.constexpr, BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_z = off_hz // H
    off_h = off_hz % H
    
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    offs_k_qk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_k_v = tl.arange(0, BLOCK_DMODEL_V)
    
    mask_qk = offs_k_qk < r
    
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_k_qk[None, :] * stride_qk
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_k_v[None, :] * stride_on
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=tl.float32)
    
    mask_q = (offs_m[:, None] < N_CTX) & mask_qk[None, :]
    q = tl.load(q_ptrs, mask=mask_q, other=0.0)

    if SPLIT_K == 1:
        # Unmasked blocks below diagonal
        for start_n in range(0, start_m * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n_curr = start_n + offs_n
            
            k_ptrs = K + k_offset + offs_n_curr[None, :] * stride_kn + offs_k_qk[:, None] * stride_kk
            v_ptrs = V + v_offset + offs_n_curr[:, None] * stride_vn + offs_k_v[None, :] * stride_vk
            
            k = tl.load(k_ptrs, mask=mask_qk[:, None], other=0.0)
            
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k, allow_tf32=True)
            qk = qk * sm_scale
            
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_ij[:, None])
            
            l_ij = tl.sum(p, 1)
            alpha = tl.exp(m_i - m_ij)
            
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            
            v = tl.load(v_ptrs)
            p = p.to(v.dtype)
            
            acc += tl.dot(p, v, allow_tf32=True)
            m_i = m_ij
            
        # Masked diagonal block
        for start_n in range(start_m * BLOCK_M, (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n_curr = start_n + offs_n
            
            k_ptrs = K + k_offset + offs_n_curr[None, :] * stride_kn + offs_k_qk[:, None] * stride_kk
            v_ptrs = V + v_offset + offs_n_curr[:, None] * stride_vn + offs_k_v[None, :] * stride_vk
            
            mask_k = (offs_n_curr[None, :] < N_CTX) & mask_qk[:, None]
            k = tl.load(k_ptrs, mask=mask_k, other=0.0)
            
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k, allow_tf32=True)
            qk = qk * sm_scale
            
            # causal mask
            qk = tl.where(offs_m[:, None] >= offs_n_curr[None, :], qk, float("-inf"))
            
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            
            m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
            p = tl.exp(qk - m_ij_safe[:, None])
            
            l_ij = tl.sum(p, 1)
            
            m_i_safe = tl.where(m_i == float("-inf"), 0.0, m_i)
            alpha = tl.exp(m_i_safe - m_ij_safe)
            
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            
            v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX, other=0.0)
            p = p.to(v.dtype)
            
            acc += tl.dot(p, v, allow_tf32=True)
            m_i = m_ij
            
        acc = acc / l_i[:, None]
        
        mask_o = offs_m[:, None] < N_CTX
        mask_m = offs_m < N_CTX
        
        mix_g_ptrs = mix_g_ptr + off_z * stride_mix_g_b + off_h * stride_mix_g_h + offs_m * stride_mix_g_n
        mix_g_val = tl.load(mix_g_ptrs, mask=mask_m, other=0.0)
        
        acc = acc * mix_g_val[:, None]
        
        # IN-PLACE ADDITION to the output initialized by GLA
        prev_out = tl.load(o_ptrs, mask=mask_o, other=0.0)
        new_out = prev_out + acc.to(Out.dtype.element_ty)
        tl.store(o_ptrs, new_out, mask=mask_o)
    else:
        split_k_id = tl.program_id(2)
        for block_idx in range(split_k_id, tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N), SPLIT_K):
            start_n = block_idx * BLOCK_N
            offs_n_curr = start_n + offs_n
            
            k_ptrs = K + k_offset + offs_n_curr[None, :] * stride_kn + offs_k_qk[:, None] * stride_kk
            v_ptrs = V + v_offset + offs_n_curr[:, None] * stride_vn + offs_k_v[None, :] * stride_vk
            
            mask_k = (offs_n_curr[None, :] < N_CTX) & mask_qk[:, None]
            k = tl.load(k_ptrs, mask=mask_k, other=0.0)
            
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k, allow_tf32=True)
            qk = qk * sm_scale
            
            # Triton 3.2.0 workaround: avoid if/else branch inside loop to prevent SSA scheduling errors
            is_unmasked = start_n < start_m * BLOCK_M
            mask_safe = (offs_m[:, None] >= offs_n_curr[None, :]) | is_unmasked
            qk = tl.where(mask_safe, qk, float("-inf"))
            
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
            p = tl.exp(qk - m_ij_safe[:, None])
            
            l_ij = tl.sum(p, 1)
            
            m_i_safe = tl.where(m_i == float("-inf"), 0.0, m_i)
            alpha = tl.exp(m_i_safe - m_ij_safe)
            
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            
            v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX, other=0.0)
            p = p.to(v.dtype)
            
            acc += tl.dot(p, v, allow_tf32=True)
            m_i = m_ij
            
        part_out_ptrs = Part_Out_ptr + off_hz * stride_part_out_zh + split_k_id * stride_part_out_k + offs_m[:, None] * stride_part_out_n + tl.arange(0, BLOCK_DMODEL_V)[None, :] * stride_part_out_d
        part_l_ptrs = Part_L_ptr + off_hz * stride_part_l_zh + split_k_id * stride_part_l_k + offs_m * stride_part_l_n
        part_m_ptrs = Part_M_ptr + off_hz * stride_part_m_zh + split_k_id * stride_part_m_k + offs_m * stride_part_m_n
        
        mask_m = offs_m < N_CTX
        tl.store(part_out_ptrs, acc, mask=mask_m[:, None])
        tl.store(part_l_ptrs, l_i, mask=mask_m)
        tl.store(part_m_ptrs, m_i, mask=mask_m)

@triton.jit
def _red_kernel(
    Part_Out_ptr, Part_L_ptr, Part_M_ptr,
    Out, mix_g_ptr,
    stride_part_out_zh, stride_part_out_k, stride_part_out_n, stride_part_out_d,
    stride_part_l_zh, stride_part_l_k, stride_part_l_n,
    stride_part_m_zh, stride_part_m_k, stride_part_m_n,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_mix_g_b, stride_mix_g_h, stride_mix_g_n,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL_V: tl.constexpr,
    SPLIT_K: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_z = off_hz // H
    off_h = off_hz % H
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N_CTX
    
    m_global = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    
    # 1. Compute global maximum m_global across all split segments
    for k in range(SPLIT_K):
        part_m_ptrs = Part_M_ptr + off_hz * stride_part_m_zh + k * stride_part_m_k + offs_m * stride_part_m_n
        m_k = tl.load(part_m_ptrs, mask=mask_m, other=-float("inf"))
        m_global = tl.maximum(m_global, m_k)
        
    # 2. Rescale partial outputs and accumulate global denominator and values
    l_global = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_global = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=tl.float32)
    
    for k in range(SPLIT_K):
        part_m_ptrs = Part_M_ptr + off_hz * stride_part_m_zh + k * stride_part_m_k + offs_m * stride_part_m_n
        m_k = tl.load(part_m_ptrs, mask=mask_m, other=-float("inf"))
        
        part_l_ptrs = Part_L_ptr + off_hz * stride_part_l_zh + k * stride_part_l_k + offs_m * stride_part_l_n
        l_k = tl.load(part_l_ptrs, mask=mask_m, other=0.0)
        
        part_out_ptrs = Part_Out_ptr + off_hz * stride_part_out_zh + k * stride_part_out_k + offs_m[:, None] * stride_part_out_n + tl.arange(0, BLOCK_DMODEL_V)[None, :] * stride_part_out_d
        acc_k = tl.load(part_out_ptrs, mask=mask_m[:, None], other=0.0)
        
        m_k_safe = tl.where(m_k == float("-inf"), 0.0, m_k)
        m_global_safe = tl.where(m_global == float("-inf"), 0.0, m_global)
        alpha = tl.exp(m_k_safe - m_global_safe)
        
        # If m_k was -inf, alpha doesn't matter because l_k and acc_k are 0
        l_global += l_k * alpha
        acc_global += acc_k * alpha[:, None]
        
    l_global_safe = tl.where(l_global > 0.0, l_global, 1.0)
    acc_global = acc_global / l_global_safe[:, None]
    
    # 3. Apply mix gate and write to Out
    mix_g_ptrs = mix_g_ptr + off_z * stride_mix_g_b + off_h * stride_mix_g_h + offs_m * stride_mix_g_n
    mix_g_val = tl.load(mix_g_ptrs, mask=mask_m, other=0.0)
    acc_global = acc_global * mix_g_val[:, None]
    
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + tl.arange(0, BLOCK_DMODEL_V)[None, :] * stride_on
    
    prev_out = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    new_out = prev_out + acc_global.to(Out.dtype.element_ty)
    tl.store(o_ptrs, new_out, mask=mask_m[:, None])

def exact_attention_triton(q, k, v, r, sm_scale, mix_g, out=None, split_k=None):
    Z, H, N_CTX, D_qk = q.shape
    D_v = v.shape[-1]
    
    if out is None:
        out = torch.zeros_like(v)
    
    BLOCK_DMODEL_V = triton.next_power_of_2(D_v)
    BLOCK_DMODEL_QK = max(triton.next_power_of_2(r), 16)
    
    max_sram = get_device_max_sram(q.device)
    
    # Bucket N_CTX to prevent constant recompilation during variable-length inference prefill,
    # while still allowing autotuning to choose optimal configs for different context scales.
    n_ctx_bucket = get_bucket(N_CTX)
    
    if split_k is None:
        # Compute how much VRAM Split-K partials would cost, and only
        # enable it if the buffers fit comfortably (< 25% of free VRAM).
        free_vram, _ = torch.cuda.mem_get_info(q.device)
        vram_budget = free_vram // 4  # 25% of free VRAM
        
        # Per-split partial cost: N_CTX * (D_v*4 + 4 + 4) bytes  (out + l + m in fp32)
        per_split_bytes = N_CTX * (BLOCK_DMODEL_V * 4 + 4 + 4) * Z * H
        
        # Try split_k = 4, then 2, then 1
        split_k = 1
        for candidate in [4, 2]:
            if candidate * per_split_bytes <= vram_budget:
                split_k = candidate
                break
            
    if split_k > 1:
        # Allocate per-token partials for each split segment
        part_out = torch.zeros((Z * H, split_k, N_CTX, BLOCK_DMODEL_V), dtype=torch.float32, device=q.device)
        part_l = torch.zeros((Z * H, split_k, N_CTX), dtype=torch.float32, device=q.device)
        part_m = torch.full((Z * H, split_k, N_CTX), float('-inf'), dtype=torch.float32, device=q.device)
        
        grid = lambda meta: (triton.cdiv(N_CTX, meta['BLOCK_M']), Z * H, split_k)
        
        _fwd_kernel[grid](
            q, k, v, out, mix_g,
            sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            mix_g.stride(0), mix_g.stride(1), mix_g.stride(2),
            Z, H, N_CTX, r, n_ctx_bucket,
            part_out, part_l, part_m,
            part_out.stride(0), part_out.stride(1), part_out.stride(2), part_out.stride(3),
            part_l.stride(0), part_l.stride(1), part_l.stride(2),
            part_m.stride(0), part_m.stride(1), part_m.stride(2),
            BLOCK_DMODEL_QK=BLOCK_DMODEL_QK,
            BLOCK_DMODEL_V=BLOCK_DMODEL_V,
            SPLIT_K=split_k
        )
        
        # Retrieve chosen BLOCK_M or use fallback
        best_block_m = 128
        if hasattr(_fwd_kernel, 'best_config') and _fwd_kernel.best_config is not None:
            best_block_m = _fwd_kernel.best_config.kwargs.get('BLOCK_M', 128)
            
        grid_red = (triton.cdiv(N_CTX, best_block_m), Z * H)
        _red_kernel[grid_red](
            part_out, part_l, part_m,
            out, mix_g,
            part_out.stride(0), part_out.stride(1), part_out.stride(2), part_out.stride(3),
            part_l.stride(0), part_l.stride(1), part_l.stride(2),
            part_m.stride(0), part_m.stride(1), part_m.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            mix_g.stride(0), mix_g.stride(1), mix_g.stride(2),
            Z, H, N_CTX,
            BLOCK_M=best_block_m,
            BLOCK_DMODEL_V=BLOCK_DMODEL_V,
            SPLIT_K=split_k
        )
    else:
        grid = lambda meta: (triton.cdiv(N_CTX, meta['BLOCK_M']), Z * H)
        
        # Pass dummy tensors and 0 strides when SPLIT_K = 1
        _fwd_kernel[grid](
            q, k, v, out, mix_g,
            sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            mix_g.stride(0), mix_g.stride(1), mix_g.stride(2),
            Z, H, N_CTX, r, n_ctx_bucket,
            out, out, out,
            0, 0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            BLOCK_DMODEL_QK=BLOCK_DMODEL_QK,
            BLOCK_DMODEL_V=BLOCK_DMODEL_V,
            SPLIT_K=1
        )
        
    return out
