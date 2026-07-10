"""Debug script v2: Zoom into the RMSNorm divergence."""
import torch
import torch.nn.functional as F

# ── Monkey-patch ──────────────────────────────────────────────────────
import src.hofa_decode_triton as _dm
import src.HybridOutlierFactorizedAttention as _am

_orig = _dm.fused_hofa_decode
_cap = {}

def _hook(q, k, v, cache_k, cache_v, state_in, state_out,
          log_gamma, mix_g, norm_w, R, seq_len, sm_scale):
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
    _cap['Y']         = result.clone()
    _cap['state_out'] = state_out.clone()
    return result

_am.fused_hofa_decode = _hook

from src.HybridOutlierFactorizedAttentionTrain import (
    HybridOutlierFactorizedAttention as HOFA_Train)
from src.HybridOutlierFactorizedAttention import (
    HybridOutlierFactorizedAttention as HOFA_Infer)
from src.modules.modules import apply_rotary_pos_emb
from src.config import ModelConfig
from training.training_config import LanguageModelingExperimentConfig

def pm(name, a, b):
    af, bf = a.float().flatten(), b.float().flatten()
    cos = F.cosine_similarity(af[None], bf[None]).item()
    md  = (af - bf).abs().max().item()
    tag = "✓" if cos > 0.99 else "✗"
    print(f"  {tag} {name:<45s} CosSim={cos:.6f}  MaxDiff={md:.4e}")

def debug_decode(config: LanguageModelingExperimentConfig):
    dev = torch.device("cuda")
    cfg = config.model_config
    cfg.r = cfg.r if isinstance(cfg.r, int) else cfg.r[0] # for simplicity
    H, D, R, J = cfg.num_heads, cfg.d_model//cfg.num_heads, cfg.r, (cfg.d_model//cfg.num_heads)-cfg.r

    torch.manual_seed(42)
    train = HOFA_Train(cfg).to(dev)
    infer = HOFA_Infer(cfg).to(dev)
    infer.load_state_dict(train.state_dict())
    train.eval(); infer.eval()

    B, N = 2, 32
    torch.manual_seed(100)
    x = torch.randn(B, N, cfg.d_model, device=dev, dtype=torch.bfloat16)

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        # ═══ Prefill ═════════════════════════════════════════════════
        from fla.ops.gla import chunk_gla
        sf = D**0.25
        Q = (train.W_q(x)/sf).view(B,N,H,D).transpose(1,2)
        K = (train.W_k(x)/sf).view(B,N,H,D).transpose(1,2)
        V = train.W_v(x).view(B,N,H,D).transpose(1,2)
        gl, _ = train._compute_gates_optimized(Q, K)
        lg = F.logsigmoid(-gl)
        QJ, KJ = Q[...,R:], K[...,R:]
        qt = QJ.transpose(1,2).bfloat16().contiguous()
        kt = KJ.transpose(1,2).bfloat16().contiguous()
        vt = V.transpose(1,2).bfloat16().contiguous()
        gt = lg.transpose(1,2).expand(-1,-1,-1,KJ.shape[-1]).bfloat16().contiguous()
        _, sref = chunk_gla(qt,kt,vt,g=gt,scale=1.0,output_final_state=True)
        _, co_i, si_i = infer(x, return_state=True)
        QO, KO = Q[...,:R], K[...,:R]
        c, s = train.rotary_emb(N)
        QO, KO = apply_rotary_pos_emb(QO, KO, c, s)
        co_r = (KO.contiguous(), V.contiguous())
        st_r = torch.zeros(B,H,D,D, device=dev, dtype=torch.float32)
        st_r[:,:,:J,:] = sref.float()
        si_i = (si_i[0].float(), si_i[1].float())

        # ═══ Decode step ═════════════════════════════════════════════
        torch.manual_seed(200)
        xs = torch.randn(B, 1, cfg.d_model, device=dev, dtype=torch.bfloat16)
        co_r_pre = (co_r[0].clone(), co_r[1].clone())
        st_r_pre = st_r.clone()

        # Run Train forward_step
        out_t, co_r, st_r = train.forward_step(xs, co_r, st_r)

        # Run Infer forward_step (captures kernel I/O)
        _cap.clear()
        out_i, co_i, si_i = infer.forward_step(xs, co_i, si_i)

        pm("FINAL: Train vs Infer", out_t, out_i)

        # ═══ Recompute Train intermediates ═══════════════════════════
        Qd = (train.W_q(xs)/sf).view(B,1,H,D).transpose(1,2)
        Kd = (train.W_k(xs)/sf).view(B,1,H,D).transpose(1,2)
        Vd = train.W_v(xs).view(B,1,H,D).transpose(1,2)
        gl_t, mg_t = train._compute_gates_optimized(Qd, Kd)
        QJ_t = F.pad(Qd[...,R:], (0,R))
        KJ_t = F.pad(Kd[...,R:], (0,R))
        gam_t = torch.sigmoid(-gl_t).view(B,H,1,1)
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
        print(f"\n--- Train path: cast y_i to bf16, then RMSNorm ---")
        YI_t_bf16 = YI_t_fp32.to(torch.bfloat16)
        print(f"  After bf16 cast [0,0,:8] = {YI_t_bf16[0,0,:8].float().tolist()}")
        
        # RMSNorm manually in bf16 (how Train does it)
        YI_t_bf16_4d = YI_t_bf16.unsqueeze(2)  # (B,H,1,D) for RMSNorm
        YI_t_normed_train = train.inlier_norm(YI_t_bf16_4d)
        var_train = (YI_t_bf16.float()**2).mean(-1)
        rms_train = var_train.sqrt()
        print(f"  Variance (from bf16) [0,:] = {var_train[0,:].tolist()}")
        print(f"  RMS (from bf16) [0,:]      = {rms_train[0,:].tolist()}")
        print(f"  Normed output [0,0,:8]     = {YI_t_normed_train[0,0,0,:8].float().tolist()}")

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
        pm("RMS: train(bf16) vs emu(fp32)", rms_train, rms_emu)
        pm("Normed (no scale): train vs emu", YI_t_normed_train.squeeze(2).float(), yi_normed_emu)

        # Now apply gla_scale
        nw = train.gla_scale.to(torch.bfloat16)
        YI_t_scaled = (YI_t_normed_train * nw).squeeze(2).float()
        yi_scaled_emu = yi_normed_emu * _cap['norm_w'].float()
        pm("After gla_scale: train vs emu", YI_t_scaled, yi_scaled_emu)

        # Also check: what if we DON'T cast to bf16 in Train path?
        print(f"\n--- What if Train also used fp32 for RMSNorm? ---")
        var_fix = (YI_t_fp32**2).mean(-1, keepdim=True)
        rsq_fix = torch.rsqrt(var_fix + 1e-5)
        yi_normed_fix = YI_t_fp32 * rsq_fix
        pm("Normed (fp32 fix): train_fix vs emu", yi_normed_fix, yi_normed_emu)

        # What if kernel casts to bf16 first?
        print(f"\n--- What if kernel cast y_i to bf16 before norm? ---")
        yi_e_bf16 = yi_e.to(torch.bfloat16).float()
        var_k_bf16 = (yi_e_bf16**2).mean(-1, keepdim=True)
        rsq_k_bf16 = torch.rsqrt(var_k_bf16 + 1e-5)
        yi_normed_k_bf16 = yi_e_bf16 * rsq_k_bf16
        pm("Normed (kernel bf16): vs train", YI_t_normed_train.squeeze(2).float(), yi_normed_k_bf16)

if __name__ == "__main__":
    main()
