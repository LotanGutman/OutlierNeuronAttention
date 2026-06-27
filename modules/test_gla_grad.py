import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def ref_gla(Q_J, K_J, V, gamma):
    # exact fp32 reference
    B, H, N, j = Q_J.shape
    d_head = V.shape[-1]
    Q_J = Q_J.float()
    K_J = K_J.float()
    V = V.float()
    gamma = gamma.float()
    
    Y = torch.zeros(B, H, N, d_head, dtype=torch.float32, device='cuda')
    scale = 1.0
    
    for b in range(B):
        for h in range(H):
            S = torch.zeros((j, d_head), dtype=torch.float32, device='cuda')
            for t in range(N):
                q = Q_J[b, h, t] * scale
                k = K_J[b, h, t]
                v = V[b, h, t]
                g = gamma[b, h, t]
                
                S = S * g + torch.outer(k, v)
                Y[b, h, t] = S.T @ q
                
    return Y.to(torch.float32), None

def test_chunk_gla_inlier_grad():
    from src.chunk_gla_inlier import ChunkGLAInlier
    torch.manual_seed(42)
    B = 2
    H = 4
    N = 1024
    d_head = 64
    r = 16
    j = d_head - r

    Q = torch.randn(B, H, N, d_head, dtype=torch.float32, device='cuda')
    K = torch.randn(B, H, N, d_head, dtype=torch.float32, device='cuda')
    V = torch.randn(B, H, N, d_head, dtype=torch.float32, device='cuda')
    
    inlier_idx = torch.arange(j, dtype=torch.long, device='cuda').expand(H, j)

    gate_logits = torch.randn(B, H, N, dtype=torch.float32, device='cuda')
    log_gamma = F.logsigmoid(-gate_logits)
    gamma = torch.exp(log_gamma)

    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)
    log_gamma.requires_grad_(True)

    y_triton = ChunkGLAInlier.apply(Q, K, V, log_gamma, inlier_idx, 64)
    loss_triton = y_triton.sum()
    loss_triton.backward()

    dQ_triton = Q.grad.clone()
    dK_triton = K.grad.clone()
    dV_triton = V.grad.clone()
    dlog_gamma_triton = log_gamma.grad.clone()

    Q.grad = None
    K.grad = None
    V.grad = None
    log_gamma.grad = None

    Q_J = Q.gather(-1, inlier_idx.unsqueeze(0).unsqueeze(2).expand(B, H, N, j))
    K_J = K.gather(-1, inlier_idx.unsqueeze(0).unsqueeze(2).expand(B, H, N, j))
    
    gamma_ref = gamma.clone().detach().requires_grad_(True)
    
    y_ref, _ = ref_gla(Q_J, K_J, V, gamma_ref)
    loss_ref = y_ref.sum()
    loss_ref.backward()

    dQ_ref = Q.grad.clone()
    dK_ref = K.grad.clone()
    dV_ref = V.grad.clone()
    
    # We computed dL/dgamma in ref, but dL/dlog_gamma = dL/dgamma * gamma
    dlog_gamma_ref = gamma_ref.grad.clone() * gamma

    def compare(name, ref, triton):
        diff = (ref - triton).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        median_diff = diff.median().item()
        rel_err = (diff / (ref.abs() + 1e-6)).mean().item()
        match = (mean_diff < 0.05) and (max_diff < 1.0)
        print(f"{name:10s} | max: {max_diff:.6f} | mean: {mean_diff:.6f} | med: {median_diff:.6f} | rel: {rel_err:.6f} | match: {match}")
        return match

    print("=== Gradient Correctness ===")
    matches = [
        compare("Y", y_ref, y_triton),
        compare("dQ", dQ_ref, dQ_triton),
        compare("dK", dK_ref, dK_triton),
        compare("dV", dV_ref, dV_triton),
        compare("dlog_gamma", dlog_gamma_ref, dlog_gamma_triton)
    ]
    
    if not all(matches):
        print(">>> SOME TESTS FAILED! <<<")
        exit(1)
    else:
        print(">>> ALL TESTS PASSED! <<<")

if __name__ == '__main__':
    test_chunk_gla_inlier_grad()

