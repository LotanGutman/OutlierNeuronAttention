import torch
import torch.nn.functional as F

from src.HybridOutlierFactorizedAttentionTrain import HybridOutlierFactorizedAttention as HOFA_Train
from src.HybridOutlierFactorizedAttention import HybridOutlierFactorizedAttention as HOFA_Infer
from src.modules.modules import apply_rotary_pos_emb

from src.config import ModelConfig

def print_metrics(name, ref, inf):
    ref_f = ref.to(torch.float32).flatten()
    inf_f = inf.to(torch.float32).flatten()
    
    max_diff = torch.max(torch.abs(ref_f - inf_f)).item()
    mae = torch.mean(torch.abs(ref_f - inf_f)).item()
    
    cos_sim = F.cosine_similarity(ref_f.unsqueeze(0), inf_f.unsqueeze(0)).item()
    
    mask = torch.abs(ref_f) > 1e-5
    if mask.sum() > 0:
        mean_rel_err = torch.mean(torch.abs(ref_f[mask] - inf_f[mask]) / torch.abs(ref_f[mask])).item()
    else:
        mean_rel_err = 0.0
        
    print(f"{name:<20} | MaxDiff: {max_diff:.4e} | MAE: {mae:.4e} | CosSim: {cos_sim:.6f} | MRE: {mean_rel_err:.4e}")

def validate_inference(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Standalone config for testing the layer math directly
    test_cfg = ModelConfig(
        d_model=128,
        num_heads=4,
        r=8,
        chunk_size=16,
        use_rope=True
    )
    
    torch.manual_seed(42)
    # Load model and cast to bfloat16
    hofa_train = HOFA_Train(test_cfg).to(device)
    hofa_infer = HOFA_Infer(test_cfg).to(device)
    
    # Copy parameters
    hofa_infer.load_state_dict(hofa_train.state_dict())
    
    hofa_train.eval()
    hofa_infer.eval()
    
    B, N = 2, 32
    torch.manual_seed(100)
    # Input in bfloat16
    x = torch.randn(B, N, test_cfg.d_model, device=device, dtype=torch.bfloat16)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # --- Prefill Phase ---
            scale_factor = hofa_train.d_head ** 0.25
            Q = (hofa_train.W_q(x) / scale_factor).view(B, N, hofa_train.num_heads, hofa_train.d_head).transpose(1, 2)
            K = (hofa_train.W_k(x) / scale_factor).view(B, N, hofa_train.num_heads, hofa_train.d_head).transpose(1, 2)
            V = hofa_train.W_v(x).view(B, N, hofa_train.num_heads, hofa_train.d_head).transpose(1, 2)
            
            gate_logits, mix_g = hofa_train._compute_gates_optimized(Q, K)
            log_gamma = F.logsigmoid(-gate_logits)
            
            Q_J = Q[..., hofa_train.r:]
            K_J = K[..., hofa_train.r:]
            
            from fla.ops.gla import chunk_gla
            
            q_gla_t = Q_J.transpose(1, 2).to(torch.bfloat16).contiguous()
            k_gla_t = K_J.transpose(1, 2).to(torch.bfloat16).contiguous()
            v_gla_t = V.transpose(1, 2).to(torch.bfloat16).contiguous()
            g_gla_t = log_gamma.transpose(1, 2).expand(-1, -1, -1, K_J.shape[-1]).to(torch.bfloat16).contiguous()
            
            Y_I_raw_ref_t, state_I_ref_final = chunk_gla(
                q_gla_t,
                k_gla_t,
                v_gla_t,
                g=g_gla_t,
                scale=1.0,
                output_final_state=True
            )
            # Transpose back
            Y_I_raw_ref = Y_I_raw_ref_t.transpose(1, 2)
            
            # 2. Infer Prefill
            Y_out_inf, cache_O_inf, state_I_inf = hofa_infer(x, return_state=True)
            
            # Compare prefill states
            print_metrics("BF16 Prefill State", state_I_ref_final, state_I_inf[0])
            
            # --- Decode Phase ---
            Q_O = Q[..., :hofa_train.r]
            K_O = K[..., :hofa_train.r]
            if hasattr(hofa_train, 'rotary_emb'):
                cos, sin = hofa_train.rotary_emb(N)
                Q_O, K_O = apply_rotary_pos_emb(Q_O, K_O, cos, sin)
            cache_O_ref = (K_O.contiguous(), V.contiguous())
            
            # Prepare padded state for Train
            state_I_ref_padded = torch.zeros(B, hofa_train.num_heads, hofa_train.d_head, hofa_train.d_head, device=device, dtype=torch.float32)
            state_I_ref_padded[:, :, :hofa_train.j, :] = state_I_ref_final.to(torch.float32)
            
            # Convert state_I_inf to float32 if not already (it is a tuple)
            state_I_inf = (state_I_inf[0].to(torch.float32), state_I_inf[1].to(torch.float32))
            
            for step in range(3):
                print(f"\n--- BF16 Decode Step {step+1} ---")
                x_step = torch.randn(B, 1, test_cfg.d_model, device=device, dtype=torch.bfloat16)
                
                # 1. Run Train's forward_step
                out_step_ref, cache_O_ref, state_I_ref_padded = hofa_train.forward_step(
                    x_step, cache_O_ref, state_I_ref_padded
                )
                
                # 2. Run Infer's forward_step
                out_step_inf, cache_O_inf, state_I_inf = hofa_infer.forward_step(
                    x_step, cache_O_inf, state_I_inf
                )
                
                # Compare outputs
                print_metrics("Decode Output", out_step_ref, out_step_inf)
                
                # Compare states
                state_I_ref_sliced = state_I_ref_padded[:, :, :hofa_train.j, :]
                print_metrics("Decode State", state_I_ref_sliced, state_I_inf[0])
