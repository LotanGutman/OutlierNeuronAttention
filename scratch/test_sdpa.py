import torch
import torch.nn.functional as F
import time

def test_sdpa():
    B = 4
    H = 6
    N = 32768
    device = 'cuda'
    
    Q8 = torch.randn(B, H, N, 8, dtype=torch.bfloat16, device=device)
    K8 = torch.randn(B, H, N, 8, dtype=torch.bfloat16, device=device)
    V64 = torch.randn(B, H, N, 64, dtype=torch.bfloat16, device=device)
    
    Q64 = F.pad(Q8, (0, 56))
    K64 = F.pad(K8, (0, 56))
    
    # Warmup
    for _ in range(5):
        _ = F.scaled_dot_product_attention(Q8, K8, V64, is_causal=True)
        
    torch.cuda.synchronize()
    start = time.perf_counter()
    steps = 10
    for _ in range(steps):
        _ = F.scaled_dot_product_attention(Q8, K8, V64, is_causal=True)
    torch.cuda.synchronize()
    print(f"Unpadded (d_k=8, d_v=64) time: {(time.perf_counter() - start)/steps*1000:.2f} ms")
    
    # Warmup
    for _ in range(5):
        _ = F.scaled_dot_product_attention(Q64.contiguous(), K64.contiguous(), V64.contiguous(), is_causal=True)
        
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            _ = F.scaled_dot_product_attention(Q64.contiguous(), K64.contiguous(), V64.contiguous(), is_causal=True)
    torch.cuda.synchronize()
    print(f"Padded + Contiguous + Force Flash time: {(time.perf_counter() - start)/steps*1000:.2f} ms")

if __name__ == '__main__':
    test_sdpa()
