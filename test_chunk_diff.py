import torch
from scratch.flash_linear_attention.fla.ops.gla.chunk import chunk_gla
import sys

# We need to make sure fla is accessible
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scratch', 'flash-linear-attention'))

from fla.ops.gla.chunk import chunk_gla

B, H, N, j, d_head = 4, 6, 128, 56, 64
torch.manual_seed(42)
Q = torch.randn(B, H, N, j, device='cuda', dtype=torch.float32)
K = torch.randn(B, H, N, j, device='cuda', dtype=torch.float32)
V = torch.randn(B, H, N, d_head, device='cuda', dtype=torch.float32)
g = torch.randn(B, H, N, j, device='cuda', dtype=torch.float32)
scale = d_head ** -0.5

# chunk_gla uses BT=64 by default? Let's check output difference across different settings.
o1, _ = chunk_gla(Q, K, V, g, scale=scale)
print('Done!')
