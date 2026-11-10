import torch
import triton
import triton.language as tl
from src.modules.triton_utils import get_device_max_sram, get_exact_attn_block_sizes, get_bucket, BUCKETS

def exact_attn_early_prune(configs, named_args, **kwargs):
    device = named_args['Q'].device
    max_sram = get_device_max_sram(device)
    
    block_dmodel_v = named_args.get('BLOCK_DMODEL_V', kwargs.get('BLOCK_DMODEL_V', 128))
    block_dmodel_qk = named_args.get('BLOCK_DMODEL_QK', kwargs.get('BLOCK_DMODEL_QK', 16))
    
    pruned_configs = []
    for config in configs:
        block_m = config.kwargs['BLOCK_M']
        block_n = config.kwargs['BLOCK_N']
        stages = getattr(config, 'num_stages', 2)
        # Shared memory holds Q tile and multi-staged K and V tiles
        sram_q = block_m * block_dmodel_qk * 2
        sram_kv = stages * (block_n * block_dmodel_qk * 2 + block_n * block_dmodel_v * 2)
        est = sram_q + sram_kv
        if est <= max_sram:
            pruned_configs.append(config)
    return pruned_configs

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=2, num_stages=3),
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
    BLOCK_M: tl.constexpr, BLOCK_DMODEL_QK: tl.constexpr, BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Reverse BLOCK_M order so long causal blocks dispatch first and eliminate tail wave stalls
    start_m = (tl.num_programs(0) - 1) - tl.program_id(0)
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

    # Base-2 scaled factor for fast hardware tl.math.exp2
    qk_scale = sm_scale * 1.4426950408889634

    k_ptrs = K + k_offset + offs_n[None, :] * stride_kn + offs_k_qk[:, None] * stride_kk
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_k_v[None, :] * stride_vk

    # Unmasked blocks below diagonal (causal loop peeling)
    for start_n in range(0, start_m * BLOCK_M, BLOCK_N):
        k = tl.load(k_ptrs, mask=mask_qk[:, None], other=0.0)
        
        qk = tl.dot(q, k, allow_tf32=True) * qk_scale
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        v = tl.load(v_ptrs)
        acc = tl.dot(p.to(v.dtype), v, acc, allow_tf32=True)
        m_i = m_ij

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        
    # Masked diagonal block
    for start_n in range(start_m * BLOCK_M, (start_m + 1) * BLOCK_M, BLOCK_N):
        offs_n_curr = start_n + offs_n
        
        mask_k = (offs_n_curr[None, :] < N_CTX) & mask_qk[:, None]
        k = tl.load(k_ptrs, mask=mask_k, other=0.0)
        
        qk = tl.dot(q, k, allow_tf32=True) * qk_scale
        
        # causal mask
        qk = tl.where(offs_m[:, None] >= offs_n_curr[None, :], qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        
        m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        p = tl.math.exp2(qk - m_ij_safe[:, None])
        
        l_ij = tl.sum(p, 1)
        
        m_i_safe = tl.where(m_i == float("-inf"), 0.0, m_i)
        alpha = tl.math.exp2(m_i_safe - m_ij_safe)
        
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc, allow_tf32=True)
        m_i = m_ij

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        
    acc = acc / l_i[:, None]
    
    mask_o = offs_m[:, None] < N_CTX
    mask_m = offs_m < N_CTX
    
    mix_g_ptrs = mix_g_ptr + off_z * stride_mix_g_b + off_h * stride_mix_g_h + offs_m * stride_mix_g_n
    mix_g_val = tl.load(mix_g_ptrs, mask=mask_m, other=0.0)
    
    acc = acc * mix_g_val[:, None]
    
    # In-place addition to the output initialized by GLA
    prev_out = tl.load(o_ptrs, mask=mask_o, other=0.0)
    new_out = prev_out + acc.to(Out.dtype.element_ty)
    tl.store(o_ptrs, new_out, mask=mask_o)

def exact_attention_triton(q, k, v, r, sm_scale, mix_g, out=None, split_k=None):
    Z, H, N_CTX, D_qk = q.shape
    D_v = v.shape[-1]
    
    if out is None:
        out = torch.zeros_like(v)
    
    BLOCK_DMODEL_V = triton.next_power_of_2(D_v)
    BLOCK_DMODEL_QK = max(triton.next_power_of_2(r), 16)
    
    n_ctx_bucket = get_bucket(N_CTX)
    grid = lambda meta: (triton.cdiv(N_CTX, meta['BLOCK_M']), Z * H)
    
    _fwd_kernel[grid](
        q, k, v, out, mix_g,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        mix_g.stride(0), mix_g.stride(1), mix_g.stride(2),
        Z, H, N_CTX, r, n_ctx_bucket,
        BLOCK_DMODEL_QK=BLOCK_DMODEL_QK,
        BLOCK_DMODEL_V=BLOCK_DMODEL_V,
    )
    return out
