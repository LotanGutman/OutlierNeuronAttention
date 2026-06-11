import torch
from src.chunk_gla_inlier import ChunkGLAInlier

def ref_gla(Q_J, K_J, V, gamma):
    from fla.ops.gla.chunk import chunk_gla
    gamma_J = gamma.unsqueeze(-1).expand_as(Q_J)
    g_J = torch.log2(gamma_J.clamp_min(1e-6))
    scale = V.shape[-1] ** -0.5
    return chunk_gla(Q_J.contiguous(), K_J.contiguous(), V.contiguous(), g_J.contiguous(), scale=scale)[0]

def test():
    print("=== Running Custom GLA Inlier Gradient Correctness Check ===")
    B, H, N, j, d_head = 4, 6, 128, 56, 64
    device = 'cuda'
    dtype = torch.float32

    # Fixed seeds for reproducibility
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, d_head, device=device, dtype=dtype).requires_grad_(True)
    K = torch.randn(B, H, N, d_head, device=device, dtype=dtype).requires_grad_(True)
    V = torch.randn(B, H, N, d_head, device=device, dtype=dtype).requires_grad_(True)
    gamma = torch.sigmoid(torch.randn(B, H, N, device=device, dtype=dtype)).requires_grad_(True)
    
    in_idx = torch.arange(j, dtype=torch.long, device=device).unsqueeze(0).expand(H, j)
    inlier_idx = in_idx.view(1, H, 1, j).expand(B, H, N, j)
    
    # 1. Run reference (gather + FLA chunk_gla)
    Q_J = Q.gather(-1, inlier_idx)
    K_J = K.gather(-1, inlier_idx)
    Y_ref = ref_gla(Q_J, K_J, V, gamma)
    
    dY = torch.randn_like(Y_ref)
    loss_ref = (Y_ref * dY).sum()
    loss_ref.backward()
    
    Q_grad_ref = Q.grad.clone()
    K_grad_ref = K.grad.clone()
    V_grad_ref = V.grad.clone()
    gamma_grad_ref = gamma.grad.clone()
    
    Q.grad.zero_()
    K.grad.zero_()
    V.grad.zero_()
    gamma.grad.zero_()
    
    # 2. Run Triton pseudo-fused
    Y_triton = ChunkGLAInlier.apply(Q, K, V, gamma, in_idx, 32)
    print(f"Y_ref max: {Y_ref.abs().max():.6f}, Y_triton max: {Y_triton.abs().max():.6f}")
    
    # Check max diff in first chunk
    chunk_size = 32
    diff_c0 = (Y_ref[:, :, :chunk_size, :] - Y_triton[:, :, :chunk_size, :]).abs().max().item()
    diff_c1 = (Y_ref[:, :, chunk_size:2*chunk_size, :] - Y_triton[:, :, chunk_size:2*chunk_size, :]).abs().max().item()
    diff_c2 = (Y_ref[:, :, 2*chunk_size:3*chunk_size, :] - Y_triton[:, :, 2*chunk_size:3*chunk_size, :]).abs().max().item()
    print(f"Diff chunk 0: {diff_c0:.6f}")
    print(f"Diff chunk 1: {diff_c1:.6f}")
    print(f"Diff chunk 2: {diff_c2:.6f}")
    
    loss_triton = (Y_triton * dY).sum()
    loss_triton.backward()
    
    forward_ok = torch.allclose(Y_triton, Y_ref, rtol=1e-3, atol=1e-3)
    dq_ok = torch.allclose(Q.grad, Q_grad_ref, rtol=1e-3, atol=1e-3)
    dk_ok = torch.allclose(K.grad, K_grad_ref, rtol=1e-3, atol=1e-3)
    dv_ok = torch.allclose(V.grad, V_grad_ref, rtol=1e-3, atol=1e-3)
    dgamma_ok = torch.allclose(gamma.grad, gamma_grad_ref, rtol=1e-3, atol=1e-3)
    
    print(f"Forward match: {forward_ok} (Max diff: {(Y_triton - Y_ref).abs().max().item():.6f})")
    print(f"dQ match:      {dq_ok} (Max diff: {(Q.grad - Q_grad_ref).abs().max().item():.6f})")
    print(f"dK match:      {dk_ok} (Max diff: {(K.grad - K_grad_ref).abs().max().item():.6f})")
    print(f"dV match:      {dv_ok} (Max diff: {(V.grad - V_grad_ref).abs().max().item():.6f})")
    print(f"dgamma match:  {dgamma_ok} (Max diff: {(gamma.grad - gamma_grad_ref).abs().max().item():.6f})")
    
    if forward_ok and dq_ok and dk_ok and dv_ok and dgamma_ok:
        print(">>> ALL TESTS PASSED SUCCESSFULLY! <<<")
    else:
        print(">>> SOME TESTS FAILED! <<<")

if __name__ == "__main__":
    test()
