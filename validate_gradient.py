import torch, math, sys
sys.path.append('.')
from src.config import ModelConfig, TrainingConfig
from src.torch_model import HybridOutlierFactorizedAttention as TorchHOFA
from src.triton_model import HybridOutlierFactorizedAttention as TritonHOFA

def main():
    train_cfg = TrainingConfig()
    device = torch.device(train_cfg.device)
    torch.manual_seed(train_cfg.seed)

    # Small config for fast comparison
    model_cfg = ModelConfig(d_model=64, num_heads=1, num_layers=1,
                            r=4, m=16, m_O=32, chunk_size=16, block_size=64)
    B, N = 1, 48
    x = torch.randn(B, N, model_cfg.d_model, device=device)

    # Build both models, copy weights
    torch_of = TorchHOFA(model_cfg).to(device).train()
    triton_of = TritonHOFA(model_cfg).to(device).train()
    triton_of.load_state_dict(torch_of.state_dict())

    # Forward
    out1 = torch_of(x)
    out2 = triton_of(x)
    
    forward_err = (out1 - out2).abs().max().item()
    forward_rel_err = forward_err / (out1.abs().max().item() + 1e-8)
    print(f"Forward pass absolute max error: {forward_err:.2e}")
    print(f"Forward pass relative max error: {forward_rel_err:.2e}")

    loss1 = out1.sum()
    loss2 = out2.sum()
    torch_of.zero_grad()
    triton_of.zero_grad()
    loss1.backward()
    loss2.backward()

    # Compare gradients
    max_rel_err = 0.0
    for (n1, p1), (n2, p2) in zip(torch_of.named_parameters(), triton_of.named_parameters()):
        if p1.grad is None or p2.grad is None:
            continue
        grad1 = p1.grad
        grad2 = p2.grad
        abs_err = (grad1 - grad2).abs().max().item()
        rel_err = abs_err / (grad1.abs().max().item() + 1e-8)
        max_rel_err = max(max_rel_err, rel_err)
        print(f"{n1}: rel err = {rel_err:.2e}")

if __name__ == '__main__':
    main()