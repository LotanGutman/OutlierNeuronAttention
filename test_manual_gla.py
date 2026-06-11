import torch
import sys
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scratch', 'flash-linear-attention'))
from fla.ops.gla.chunk import chunk_gla

B, H, N, j, d_head = 4, 6, 128, 56, 64
torch.manual_seed(42)
Q = torch.randn(B, H, N, d_head, device='cuda', dtype=torch.float32)
K = torch.randn(B, H, N, d_head, device='cuda', dtype=torch.float32)
V = torch.randn(B, H, N, d_head, device='cuda', dtype=torch.float32)
gamma = torch.sigmoid(torch.randn(B, H, N, device='cuda', dtype=torch.float32))

in_idx = torch.arange(j, dtype=torch.long, device='cuda').unsqueeze(0).expand(H, j)
inlier_idx = in_idx.view(1, H, 1, j).expand(B, H, N, j)

Q_J = Q.gather(-1, inlier_idx)
K_J = K.gather(-1, inlier_idx)

gamma_J = gamma.unsqueeze(-1).expand_as(Q_J)
g_J = torch.log2(gamma_J.clamp_min(1e-6))
scale = V.shape[-1] ** -0.5
Y_ref = chunk_gla(Q_J.contiguous(), K_J.contiguous(), V.contiguous(), g_J.contiguous(), scale=scale)[0]

def manual_gla(Q, K, V, gamma_j, scale):
    B, H, N, j = Q.shape
    d_head = V.shape[-1]
    Y = torch.zeros(B, H, N, d_head, device='cuda', dtype=torch.float32)
    S = torch.zeros(B, H, j, d_head, device='cuda', dtype=torch.float32)
    for t in range(N):
        q_t = Q[:, :, t, :] * scale
        k_t = K[:, :, t, :]
        v_t = V[:, :, t, :]
        gamma_t = gamma_j[:, :, t, :].unsqueeze(-1)
        S = S * gamma_t + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
        Y[:, :, t, :] = (q_t.unsqueeze(-1) * S).sum(dim=2)
    return Y

Y_manual = manual_gla(Q_J, K_J, V, gamma_J.clamp_min(1e-6), scale)
print(f"Manual vs chunk_gla max diff: {(Y_manual - Y_ref).abs().max().item()}")
