import torch
import torch.nn.functional as F
# import pytest
from src.hofa_bwd_kernels import HOFAAttentionFunction

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def ref_hofa(Q, K, V, gate_logits, mix_g, gla_scale, r):
    B, H, N, d_head = Q.shape
    j = d_head - r
    
    # Outlier pathway
    Q_O = Q[..., :r].float()
    K_O = K[..., :r].float()
    sm_scale = (d_head / r) ** 0.5
    
    # Causal exact attention
    attn_O = torch.matmul(Q_O, K_O.transpose(-1, -2)) * sm_scale
    mask = torch.triu(torch.full((N, N), float('-inf'), device=Q.device), diagonal=1)
    attn_O = attn_O + mask
    attn_O = F.softmax(attn_O, dim=-1)
    Y_O = torch.matmul(attn_O, V.float())
    
    # Inlier pathway
    Q_J = Q[..., r:].float()
    K_J = K[..., r:].float()
    gamma = torch.sigmoid(-gate_logits.float()) # (B, H, N, 1)
    
    Y_GLA = torch.zeros_like(V, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            S = torch.zeros((j, d_head), dtype=torch.float32, device=Q.device)
            for t in range(N):
                q = Q_J[b, h, t]
                k = K_J[b, h, t]
                v = V[b, h, t].float()
                g = gamma[b, h, t]
                
                S = S * g + torch.outer(k, v)
                Y_GLA[b, h, t] = S.T @ q
                
    # RMSNorm and scale
    rms = torch.rsqrt(Y_GLA.pow(2).sum(dim=-1, keepdim=True) / d_head + 1e-5)
    Y_I = Y_GLA * rms * gla_scale.float()
    
    # Dynamic blending
    Y_Final = mix_g.float() * Y_O + (1.0 - mix_g.float()) * Y_I
    return Y_Final

def run_test(B, H, N, d_head, r, chunk_size, gate_logits_val=0.0):
    torch.manual_seed(42)
    
    Q = torch.randn(B, H, N, d_head, dtype=torch.bfloat16, device='cuda').requires_grad_(True)
    K = torch.randn(B, H, N, d_head, dtype=torch.bfloat16, device='cuda').requires_grad_(True)
    V = torch.randn(B, H, N, d_head, dtype=torch.bfloat16, device='cuda').requires_grad_(True)
    
    if gate_logits_val == "rand":
        gate_logits = torch.randn(B, H, N, 1, dtype=torch.bfloat16, device='cuda').requires_grad_(True)
    else:
        gate_logits = torch.full((B, H, N, 1), gate_logits_val, dtype=torch.bfloat16, device='cuda').requires_grad_(True)
        
    mix_g = torch.sigmoid(torch.randn(B, H, N, 1, dtype=torch.bfloat16, device='cuda')).requires_grad_(True)
    gla_scale = torch.randn(1, H, 1, d_head, dtype=torch.bfloat16, device='cuda').requires_grad_(True)
    
    # Forward pass of Triton custom autograd function
    Y_triton = HOFAAttentionFunction.apply(Q, K, V, gate_logits, mix_g, gla_scale, r, chunk_size)
    
    # Retrieve states_in from the saved tensors
    states_in = Y_triton.grad_fn.saved_tensors[6] if Y_triton.grad_fn is not None else None
    if states_in is not None:
        print("states_in: max =", states_in.max().item(), "min =", states_in.min().item(), "mean =", states_in.mean().item())
        
    log_gamma = F.logsigmoid(-gate_logits).squeeze(-1)
    print("log_gamma in python: max =", log_gamma.max().item(), "min =", log_gamma.min().item())
        
    loss_triton = Y_triton.sum()
    loss_triton.backward()
    
    dQ_triton = Q.grad.clone()
    dK_triton = K.grad.clone()
    dV_triton = V.grad.clone()
    dgate_logits_triton = gate_logits.grad.clone()
    dmix_g_triton = mix_g.grad.clone()
    dgla_scale_triton = gla_scale.grad.clone()
    
    # Zero gradients
    Q.grad = None
    K.grad = None
    V.grad = None
    gate_logits.grad = None
    mix_g.grad = None
    gla_scale.grad = None
    
    # Forward and backward pass of FP32 reference
    Q_ref = Q.clone().detach().float().requires_grad_(True)
    K_ref = K.clone().detach().float().requires_grad_(True)
    V_ref = V.clone().detach().float().requires_grad_(True)
    gate_logits_ref = gate_logits.clone().detach().float().requires_grad_(True)
    mix_g_ref = mix_g.clone().detach().float().requires_grad_(True)
    gla_scale_ref = gla_scale.clone().detach().float().requires_grad_(True)
    
    Y_ref = ref_hofa(Q_ref, K_ref, V_ref, gate_logits_ref, mix_g_ref, gla_scale_ref, r)
    loss_ref = Y_ref.sum()
    loss_ref.backward()
    
    dQ_ref = Q_ref.grad.clone()
    dK_ref = K_ref.grad.clone()
    dV_ref = V_ref.grad.clone()
    dgate_logits_ref = gate_logits_ref.grad.clone()
    dmix_g_ref = mix_g_ref.grad.clone()
    dgla_scale_ref = gla_scale_ref.grad.clone()
    
    # Compare
    def check(name, triton_val, ref_val):
        ref_on_device = ref_val.to(triton_val.dtype).to(triton_val.device)
        diff = (triton_val - ref_on_device).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # Diagnostic printing
        abs_diff = diff
        max_idx = torch.argmax(abs_diff).item()
        flat_idx = max_idx
        
        if len(triton_val.shape) == 4:
            dim3 = flat_idx % triton_val.shape[3]
            flat_idx //= triton_val.shape[3]
            dim2 = flat_idx % triton_val.shape[2]
            flat_idx //= triton_val.shape[2]
            dim1 = flat_idx % triton_val.shape[1]
            dim0 = flat_idx // triton_val.shape[1]
            print(f"  Max diff in {name} at [{dim0},{dim1},{dim2},{dim3}]: "
                  f"triton={triton_val[dim0,dim1,dim2,dim3].item():.4f}, "
                  f"ref={ref_val[dim0,dim1,dim2,dim3].item():.4f}, "
                  f"diff={abs_diff[dim0,dim1,dim2,dim3].item():.6f}")
        elif len(triton_val.shape) == 3:
            dim2 = flat_idx % triton_val.shape[2]
            flat_idx //= triton_val.shape[2]
            dim1 = flat_idx % triton_val.shape[1]
            dim0 = flat_idx // triton_val.shape[1]
            print(f"  Max diff in {name} at [{dim0},{dim1},{dim2}]: "
                  f"triton={triton_val[dim0,dim1,dim2].item():.4f}, "
                  f"ref={ref_val[dim0,dim1,dim2].item():.4f}, "
                  f"diff={abs_diff[dim0,dim1,dim2].item():.6f}")
        
        # Use torch.testing.assert_close with tiered tolerances.
        # Forward pass: tighter. Gradients: looser due to BF16 accumulation over N.
        grad_names = {"dQ", "dK", "dV", "dgate_logits", "dmix_g", "dgla_scale"}
        if name in grad_names:
            atol, rtol = 2.0e-1, 5e-2
        else:
            atol, rtol = 5e-2, 2e-2
        
        try:
            torch.testing.assert_close(triton_val.float(), ref_on_device.float(), atol=atol, rtol=rtol)
            print(f"  {name} ✅ PASS (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")
        except AssertionError as e:
            print(f"  {name} ❌ FAIL (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")
            raise
        
    print(f"\n--- Testing B={B}, H={H}, N={N}, d_head={d_head}, r={r}, chunk_size={chunk_size}, gate_logits={gate_logits_val} ---")
    check("Y", Y_triton, Y_ref)
    check("dQ", dQ_triton, dQ_ref)
    check("dK", dK_triton, dK_ref)
    check("dV", dV_triton, dV_ref)
    check("dgate_logits", dgate_logits_triton, dgate_logits_ref)
    check("dmix_g", dmix_g_triton, dmix_g_ref)
    check("dgla_scale", dgla_scale_triton, dgla_scale_ref)

def test_hofa_bwd_correctness():
    # Test standard setup
    run_test(B=2, H=2, N=128, d_head=64, r=16, chunk_size=64, gate_logits_val="rand")
    run_test(B=1, H=4, N=256, d_head=128, r=64, chunk_size=64, gate_logits_val=1.0)

def test_hofa_bwd_decay_safety():
    # Test edge case: gate_logits is very positive (meaning gamma is close to 0, e.g. logsigmoid(-15))
    run_test(B=1, H=2, N=128, d_head=64, r=32, chunk_size=64, gate_logits_val=15.0)
    
if __name__ == '__main__':
    test_hofa_bwd_correctness()
    test_hofa_bwd_decay_safety()
    print("ALL TESTS PASSED SUCCESSFULLY!")
