import torch
import time
from src.chunk_gla_inlier import ChunkGLAInlier

def chunk_gla_pytorch(Q, K, V, gamma, chunk_size=32):
    B, H, N, j = Q.shape
    d_head = V.shape[-1]
    U = N // chunk_size
    
    Q_c = Q.view(B, H, U, chunk_size, j)
    K_c = K.view(B, H, U, chunk_size, j)
    V_c = V.view(B, H, U, chunk_size, d_head)
    gamma_c = gamma.view(B, H, U, chunk_size)
    
    g_c = torch.log(gamma_c)
    g_cumsum = torch.cumsum(g_c, dim=-1)
    g_ex = g_cumsum - g_c
    
    # mask
    idx = torch.arange(chunk_size, device=Q.device)
    diff = g_ex.unsqueeze(-1) - g_cumsum.unsqueeze(-2)
    mask = torch.exp(diff) * (idx.unsqueeze(-1) >= idx.unsqueeze(-2)).float()
    
    # Y_intra
    attn = torch.einsum('bhuqd,bhukd->bhuqk', Q_c, K_c) * mask
    Y_intra = torch.einsum('bhuqk,bhukd->bhuqd', attn, V_c)
    
    # Y_inter and state
    S = torch.zeros(B, H, j, d_head, dtype=torch.float32, device=Q.device)
    Y_inter_list = []
    
    g_cumsum_last = g_cumsum[..., -1]
    
    for u in range(U):
        # Y_inter
        y_int = torch.einsum('bhqj,bhjd->bhqd', Q_c[:, :, u], S) * torch.exp(g_ex[:, :, u]).unsqueeze(-1)
        Y_inter_list.append(y_int)
        
        # S update
        k_decay = torch.exp(g_cumsum_last[:, :, u].unsqueeze(-1) - g_cumsum[:, :, u])
        K_c_decayed = K_c[:, :, u] * k_decay.unsqueeze(-1)
        S = S * torch.exp(g_cumsum_last[:, :, u]).unsqueeze(-1).unsqueeze(-1) + torch.einsum('bhkd,bhkv->bhdv', K_c_decayed, V_c[:, :, u])
        
    Y_inter = torch.stack(Y_inter_list, dim=2)
    Y = (Y_intra + Y_inter).view(B, H, N, d_head)
    return Y

compiled_gla = torch.compile(chunk_gla_pytorch, fullgraph=True, mode="reduce-overhead")

def test():
    B = 4
    H = 6
    N = 4096
    j = 56
    d_head = 64
    chunk_size = 32
    device = torch.device("cuda")

    Q = torch.randn(B, H, N, j, dtype=torch.bfloat16, device=device, requires_grad=True)
    K = torch.randn(B, H, N, j, dtype=torch.bfloat16, device=device, requires_grad=True)
    V = torch.randn(B, H, N, d_head, dtype=torch.bfloat16, device=device, requires_grad=True)
    gamma = torch.sigmoid(torch.randn(B, H, N, dtype=torch.bfloat16, device=device)).requires_grad_(True)
    
    # compile warmup
    print("Warming up compiled PT...")
    for _ in range(3):
        Y = compiled_gla(Q, K, V, gamma)
        Y.sum().backward()
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        Y = compiled_gla(Q, K, V, gamma)
        Y.sum().backward()
    torch.cuda.synchronize()
    print(f"Compiled PT: {(time.time() - start) * 100} ms")

if __name__ == "__main__":
    test()
