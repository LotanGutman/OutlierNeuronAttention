import torch
import torch.nn.functional as F

from src.HybridOutlierFactorizedAttentionTrain import HybridOutlierFactorizedAttention as HOFA_Train
from src.HybridOutlierFactorizedAttention import HybridOutlierFactorizedAttention as HOFA_Infer
from src.modules.modules import apply_rotary_pos_emb
from src.config import ModelConfig
from training.training_config import LanguageModelingExperimentConfig

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

def pm(name, a, b):
    af, bf = a.float().flatten(), b.float().flatten()
    cos = F.cosine_similarity(af[None], bf[None]).item()
    md  = (af - bf).abs().max().item()
    tag = "✓" if cos > 0.99 else "✗"
    print(f"  {tag} {name:<45s} CosSim={cos:.6f}  MaxDiff={md:.4e}")

def validate_inference(config: LanguageModelingExperimentConfig, verbose: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    cfg = config.model_config
    cfg.r = cfg.r if isinstance(cfg.r, int) else cfg.r[0] # for simplicity
    H, D, R, J = cfg.num_heads, cfg.d_model//cfg.num_heads, cfg.r, (cfg.d_model//cfg.num_heads)-cfg.r
    
    # ── Monkey-patch for deep dive if verbose ──────────────────────────
    _cap = {}
    if verbose:
        import src.hofa_decode_triton as _dm
        import src.HybridOutlierFactorizedAttention as _am
        _orig = _dm.fused_hofa_decode
        
        def _hook(q, k, v, cache_k, cache_v, state_in, state_out,
                  log_gamma, mix_g, norm_w, R, seq_len, sm_scale):
            if 'q' not in _cap:
                _cap['q']         = q.clone()
                _cap['k']         = k.clone()
                _cap['v']         = v.clone()
                _cap['cache_k']   = cache_k.clone()
                _cap['cache_v']   = cache_v.clone()
                _cap['state_in']  = state_in.clone()
                _cap['log_gamma'] = log_gamma.clone()
                _cap['mix_g']     = mix_g.clone()
                _cap['norm_w']    = norm_w.clone()
                _cap['R']         = R
                _cap['seq_len']   = seq_len
                _cap['sm_scale']  = sm_scale
            result = _orig(q, k, v, cache_k, cache_v, state_in, state_out,
                           log_gamma, mix_g, norm_w, R, seq_len, sm_scale)
            if 'Y' not in _cap:
                _cap['Y']         = result.clone()
                _cap['state_out'] = state_out.clone()
            return result

        _am.fused_hofa_decode = _hook

    torch.manual_seed(42)
    # Load model
    hofa_train = HOFA_Train(cfg).to(device)
    hofa_infer = HOFA_Infer(cfg).to(device)
    
    # Copy parameters
    hofa_infer.load_state_dict(hofa_train.state_dict())
    
    hofa_train.eval()
    hofa_infer.eval()
    
    B, N = 2, 32
    torch.manual_seed(100)
    # Input in bfloat16
    x = torch.randn(B, N, cfg.d_model, device=device, dtype=torch.bfloat16)
    
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
            
            out_step_ref = None
            out_step_inf = None

            for step in range(3):
                print(f"\n--- BF16 Decode Step {step+1} ---")
                torch.manual_seed(200 + step)
                x_step = torch.randn(B, 1, cfg.d_model, device=device, dtype=torch.bfloat16)
                
                # 1. Run Train's forward_step
                out_step_ref, cache_O_ref, state_I_ref_padded = hofa_train.forward_step(
                    x_step, cache_O_ref, state_I_ref_padded
                )
                
                if verbose and step == 0:
                    _cap.clear()

                # 2. Run Infer's forward_step
                out_step_inf, cache_O_inf, state_I_inf = hofa_infer.forward_step(
                    x_step, cache_O_inf, state_I_inf
                )
                
                # Compare outputs
                print_metrics("Decode Output", out_step_ref, out_step_inf)
                
                # Compare states
                state_I_ref_sliced = state_I_ref_padded[:, :, :hofa_train.j, :]
                print_metrics("Decode State", state_I_ref_sliced, state_I_inf[0])

            pm("FINAL: Train vs Infer", out_step_ref, out_step_inf)

            if verbose:
                # ═══ Recompute Train intermediates for step 1 ═══════════════════════════
                # We need to recreate the first step logic exactly to show the deep dive
                torch.manual_seed(200)
                xs = torch.randn(B, 1, cfg.d_model, device=device, dtype=torch.bfloat16)
                
                Qd = (hofa_train.W_q(xs)/scale_factor).view(B,1,H,D).transpose(1,2)
                Kd = (hofa_train.W_k(xs)/scale_factor).view(B,1,H,D).transpose(1,2)
                Vd = hofa_train.W_v(xs).view(B,1,H,D).transpose(1,2)
                gl_t, mg_t = hofa_train._compute_gates_optimized(Qd, Kd)
                QJ_t = F.pad(Qd[...,R:], (0,R))
                KJ_t = F.pad(Kd[...,R:], (0,R))
                gam_t = torch.sigmoid(-gl_t).view(B,H,1,1)
                
                # Reconstruct st_r_pre from before step 1
                st_r_pre = torch.zeros(B,H,D,D, device=device, dtype=torch.float32)
                st_r_pre[:,:,:J,:] = state_I_ref_final.to(torch.float32)

                st_new_t = st_r_pre*gam_t.float() + torch.einsum(
                    'bhd,bhm->bhdm', KJ_t.squeeze(2).float(), Vd.squeeze(2).float())
                YI_t_fp32 = torch.einsum('bhd,bhdm->bhm',
                    QJ_t.squeeze(2).float(), st_new_t)   # (B,H,D) float32

                # ═══ Emulation y_i (kernel-style, from captured inputs) ═════
                q_k = _cap['q']; k_k = _cap['k']; v_k = _cap['v']
                qJ_e = q_k[:,:,0,R:R+J].float()
                kJ_e = k_k[:,:,0,R:R+J].float()
                v_s_e = v_k[:,:,0,:].float()
                gamma_e = _cap['log_gamma'][:,:,0].float().exp()
                state_e = _cap['state_in'][:,:,:J,:D].float()
                st_new_e = gamma_e.view(B,H,1,1)*state_e + kJ_e.unsqueeze(-1)*v_s_e.unsqueeze(-2)
                yi_e = (qJ_e.unsqueeze(-1)*st_new_e).sum(-2)  # (B,H,D) float32

                print("\n" + "="*70)
                print("RMSNORM DEEP DIVE")
                print("="*70)

                pm("y_i (float32): Train vs Emu", YI_t_fp32, yi_e)

                # Show raw y_i stats
                print(f"\n  Train y_i[0,0,:8] = {YI_t_fp32[0,0,:8].tolist()}")
                print(f"  Emu   y_i[0,0,:8] = {yi_e[0,0,:8].tolist()}")
                print(f"  Train y_i stats: min={YI_t_fp32.min():.6e}  max={YI_t_fp32.max():.6e}  mean={YI_t_fp32.mean():.6e}  std={YI_t_fp32.std():.6e}")
                print(f"  Emu   y_i stats: min={yi_e.min():.6e}  max={yi_e.max():.6e}  mean={yi_e.mean():.6e}  std={yi_e.std():.6e}")

                # Compute RMSNorm both ways
                print(f"\n--- Train path: explicitly cast y_i to float32 before RMSNorm (FIXED) ---")
                
                YI_t_float = YI_t_fp32.float()
                YI_t_float_4d = YI_t_float.unsqueeze(2)  # (B,H,1,D) for RMSNorm
                YI_t_normed_train = hofa_train.inlier_norm(YI_t_float_4d)
                
                var_train = (YI_t_float**2).mean(-1)
                rms_train = var_train.sqrt()
                print(f"  Variance (float32) [0,:] = {var_train[0,:].tolist()}")
                print(f"  RMS (float32) [0,:]      = {rms_train[0,:].tolist()}")
                print(f"  Normed output [0,0,:8]   = {YI_t_normed_train[0,0,0,:8].float().tolist()}")

                print(f"\n--- Kernel/Emu path: RMSNorm in float32 ---")
                var_emu = (yi_e**2).mean(-1)
                rms_emu = var_emu.sqrt()
                rsq_emu = torch.rsqrt(var_emu.unsqueeze(-1) + 1e-5)
                yi_normed_emu = yi_e * rsq_emu
                print(f"  Variance (float32) [0,:]   = {var_emu[0,:].tolist()}")
                print(f"  RMS (float32) [0,:]        = {rms_emu[0,:].tolist()}")
                print(f"  Normed output [0,0,:8]     = {yi_normed_emu[0,0,:8].tolist()}")

                # Comparison
                print(f"\n--- Comparison ---")
                pm("RMS: train vs emu", rms_train, rms_emu)
                pm("Normed (no scale): train vs emu", YI_t_normed_train.squeeze(2).float(), yi_normed_emu)

                # Now apply gla_scale
                nw = hofa_train.gla_scale.to(torch.bfloat16)
                YI_t_scaled = (YI_t_normed_train * nw).squeeze(2).float()
                yi_scaled_emu = yi_normed_emu * _cap['norm_w'].float()
                pm("After gla_scale: train vs emu", YI_t_scaled, yi_scaled_emu)

                # Restore original hook
                import src.HybridOutlierFactorizedAttention as _am
                _am.fused_hofa_decode = _orig

if __name__ == "__main__":
    config = LanguageModelingExperimentConfig()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    validate_inference(config, verbose=args.verbose)
