import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out, mix_g_ptr,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_mix_g_b, stride_mix_g_h, stride_mix_g_n,
    Z, H, N_CTX, r,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL_QK: tl.constexpr, BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
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

    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
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
        p = tl.exp(qk - m_ij[:, None])
        
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        
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

def exact_attention_triton(q, k, v, r, sm_scale, mix_g, out=None):
    Z, H, N_CTX, D_qk = q.shape
    D_v = v.shape[-1]
    
    if out is None:
        out = torch.zeros_like(v)
    
    BLOCK_DMODEL_V = triton.next_power_of_2(D_v)
    BLOCK_DMODEL_QK = max(triton.next_power_of_2(r), 16)
    
    # Query actual GPU shared memory to pick safe block sizes.
    try:
        _props = triton.runtime.driver.active.utils.get_device_properties(0)
        max_sram = _props["max_shared_mem"]
    except (AttributeError, KeyError):
        max_sram = torch.cuda.get_device_properties(q.device).shared_memory_per_block_optin
    
    # Estimate peak SRAM: acc[M,V]*4 + q[M,QK]*2 + k[N,QK]*2 + v[N,V]*2 + qk[M,N]*4
    BLOCK_M = 128
    BLOCK_N = 64
    while BLOCK_M > 16:
        est = (BLOCK_M * BLOCK_DMODEL_V * 4 + BLOCK_M * BLOCK_DMODEL_QK * 2
               + BLOCK_N * BLOCK_DMODEL_QK * 2 + BLOCK_N * BLOCK_DMODEL_V * 2
               + BLOCK_M * BLOCK_N * 4)
        if est <= max_sram:
            break
        BLOCK_M //= 2
        BLOCK_N //= 2
    
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    
    _fwd_kernel[grid](
        q, k, v, out, mix_g,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        mix_g.stride(0), mix_g.stride(1), mix_g.stride(2),
        Z, H, N_CTX, r,
        BLOCK_DMODEL_QK=max(triton.next_power_of_2(r), 16),
        BLOCK_DMODEL_V=BLOCK_DMODEL_V,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2
    )
    return out
