import torch
from src.chunk_gla_inlier import ChunkGLAInlier
import time

def test_kernel():
    B = 4
    H = 6
    N = 131072
    j = 56
    d_head = 64
    chunk_size = 64
    
    device = 'cuda'
    Q = torch.randn(B, H, N, j, dtype=torch.bfloat16, device=device)
    K = torch.randn(B, H, N, j, dtype=torch.bfloat16, device=device)
    V = torch.randn(B, H, N, d_head, dtype=torch.bfloat16, device=device)
    gamma = torch.rand(B, H, N, dtype=torch.float32, device=device)
    
    # Warmup
    for _ in range(5):
        _ = ChunkGLAInlier.apply(Q, K, V, gamma, chunk_size)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    steps = 10
    for _ in range(steps):
        _ = ChunkGLAInlier.apply(Q, K, V, gamma, chunk_size)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    print(f"Kernel time at 128k: {(end - start) / steps * 1000:.2f} ms")

if __name__ == '__main__':
    test_kernel()
