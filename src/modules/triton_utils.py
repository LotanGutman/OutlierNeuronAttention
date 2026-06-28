import torch
import triton

def get_device_max_sram(device=None) -> int:
    if device is None:
        device = torch.cuda.current_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    if isinstance(device, torch.device):
        device_idx = device.index if device.index is not None else 0
    else:
        device_idx = device
        
    try:
        _props = triton.runtime.driver.active.utils.get_device_properties(device_idx)
        max_sram = _props["max_shared_mem"]
    except Exception:
        max_sram = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    return max_sram

def get_exact_attn_block_sizes(r, d_v, max_sram, default_m=128, default_n=64) -> (int, int):
    BLOCK_DMODEL_V = triton.next_power_of_2(d_v)
    BLOCK_DMODEL_QK = max(triton.next_power_of_2(r), 16)
    
    BLOCK_M = default_m
    BLOCK_N = default_n
    while BLOCK_M > 16:
        est = (BLOCK_M * BLOCK_DMODEL_V * 4 + BLOCK_M * BLOCK_DMODEL_QK * 2
               + BLOCK_N * BLOCK_DMODEL_QK * 2 + BLOCK_N * BLOCK_DMODEL_V * 2
               + BLOCK_M * BLOCK_N * 4)
        if est <= max_sram:
            break
        BLOCK_M //= 2
        BLOCK_N //= 2
    return BLOCK_M, BLOCK_N

def get_hofa_bwd_chunk_size(j_padded, d_head_padded, max_sram, initial_chunk_size=64) -> int:
    state_bytes = j_padded * d_head_padded * 4
    chunk_size = initial_chunk_size
    while chunk_size > 16:
        chunk_bytes = 8 * chunk_size**2 + (4 * j_padded + 10 * d_head_padded) * chunk_size
        if state_bytes + chunk_bytes <= max_sram:
            break
        chunk_size //= 2
    return chunk_size
