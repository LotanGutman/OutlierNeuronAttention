import torch
from src.chunk_gla_inlier import ChunkGLAInlier

def ref_gla(Q, K, V, gamma):
    B, H, N, j = Q.shape
    d_head = V.shape[-1]
    
    Y = torch.zeros_like(V)
    state = torch.zeros(B, H, j, d_head, dtype=torch.float32, device=Q.device)
    
    for t in range(N):
        q_t = Q[:, :, t].float()
        y_t = torch.einsum('bhj,bhjd->bhd', q_t, state)
        Y[:, :, t] = y_t.to(Q.dtype)
        
        k_t = K[:, :, t].float()
        v_t = V[:, :, t].float()
        g_t = gamma[:, :, t].unsqueeze(-1).unsqueeze(-1).float()
        state = g_t * state + torch.einsum('bhj,bhd->bhjd', k_t, v_t)
        
    return Y

def test():
    print("=== Running Custom GLA Inlier Gradient Correctness Check ===")
    B, H, N, j, d_head = 4, 6, 128, 56, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    print(f"Shape: B={B}, H={H}, N={N}, j={j}, d_head={d_head}")
    
    Q = torch.randn(B, H, N, j, device=device, dtype=torch.bfloat16).requires_grad_(True)
    K = torch.randn(B, H, N, j, device=device, dtype=torch.bfloat16).requires_grad_(True)
    V = torch.randn(B, H, N, d_head, device=device, dtype=torch.bfloat16).requires_grad_(True)
    gamma = torch.sigmoid(torch.randn(B, H, N, device=device, dtype=torch.bfloat16)).requires_grad_(True)
    
    # Run reference
    Y_ref = ref_gla(Q, K, V, gamma)
    dY = torch.randn_like(Y_ref)
    loss_ref = (Y_ref * dY).sum()
    loss_ref.backward()
    
    Q_grad_ref = Q.grad.clone()
    K_grad_ref = K.grad.clone()
    V_grad_ref = V.grad.clone()
    gamma_grad_ref = gamma.grad.clone()
    
    # Reset gradients
    Q.grad.zero_()
    K.grad.zero_()
    V.grad.zero_()
    gamma.grad.zero_()
    
    # Run Triton
    Y_triton = ChunkGLAInlier.apply(Q, K, V, gamma, 32)
    loss_triton = (Y_triton * dY).sum()
    loss_triton.backward()
    
    # Compare
    forward_ok = torch.allclose(Y_triton, Y_ref, rtol=1e-2, atol=1e-2)
    dq_ok = torch.allclose(Q.grad, Q_grad_ref, rtol=1e-2, atol=1e-2)
    dk_ok = torch.allclose(K.grad, K_grad_ref, rtol=1e-2, atol=1e-2)
    dv_ok = torch.allclose(V.grad, V_grad_ref, rtol=1e-2, atol=1e-2)
    dgamma_ok = torch.allclose(gamma.grad, gamma_grad_ref, rtol=1e-2, atol=1e-2)
    
    print(f"Forward match: {forward_ok} (Max diff: {(Y_triton.float() - Y_ref.float()).abs().max().item():.6f})")
    print(f"dQ match:      {dq_ok} (Max diff: {(Q.grad.float() - Q_grad_ref.float()).abs().max().item():.6f})")
    print(f"dK match:      {dk_ok} (Max diff: {(K.grad.float() - K_grad_ref.float()).abs().max().item():.6f})")
    print(f"dV match:      {dv_ok} (Max diff: {(V.grad.float() - V_grad_ref.float()).abs().max().item():.6f})")
    print(f"dgamma match:  {dgamma_ok} (Max diff: {(gamma.grad.float() - gamma_grad_ref.float()).abs().max().item():.6f})")
    
    all_ok = forward_ok and dq_ok and dk_ok and dv_ok and dgamma_ok
    if all_ok:
        print(">>> ALL TESTS PASSED SUCCESSFULLY! <<<")
    else:
        print(">>> SOME TESTS FAILED! <<<")
        assert False, "Gradient correctness check failed"

if __name__ == "__main__":
    test()
