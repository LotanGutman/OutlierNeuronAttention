import torch
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit

from benchmarks.benchmarks_configs import InductionDegradationExperimentConfig

def run_induction_degradation_experiment(config=None):
    from src.modules.benchmark_utils import GenericBenchmarkLM
    from benchmarks.benchmark_induction import train_induction

    if config is None:
        config = InductionDegradationExperimentConfig()
    device = config.device
    torch.manual_seed(config.seed)
    
    models_to_test = config.models_to_test
    seq_lengths = config.seq_lengths
    
    print(f"\n========================================")
    print(f" Starting r=10 Context Length Degradation Experiment")
    print(f"========================================")
    
    for seq_len in seq_lengths:
        config.seq_len = seq_len
        print(f"\n{'='*40}")
        print(f" EVALUATING SEQUENCE LENGTH (N): {seq_len}")
        print(f"{'='*40}")
        
        for name, attn_type, r_val in models_to_test:
            # Resetting the seed here ensures every model sees the *exact same* 
            # sequence of training data, providing the fairest possible comparison.
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            
            if r_val is not None:
                config.model_config.r = r_val
                
            model = GenericBenchmarkLM(
                vocab_size=config.vocab_size,
                d_model=config.model_config.d_model,
                attn_type=attn_type,
                num_heads=config.model_config.num_heads,
                num_layers=config.model_config.num_layers,
                model_cfg=config.model_config
            ).to(device)

            checkpoint_dir = f"data/induction_degradation_models/seqlen_{seq_len}/induction_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            history = train_induction(model, config, name, checkpoint_dir=checkpoint_dir)
            
            # Save final model state dict for easy loading later
            torch.save(model.state_dict(), f"{checkpoint_dir}/final_model.pt")
            
            del model
            torch.cuda.empty_cache()

def extract_measured_gamma(seq_lengths, models_to_test, r_target=10, d_h=32, num_layers=4):
    """
    Extracts the actual accumulated outlier signal shift directly from the model weight matrices:
      S = sqrt(d_h / r)
      Layer Shift = S * (1 / sqrt(d_h)) * sum_{k=1}^r ||WQ_k||_2 * ||WK_k||_2
      mu_weights = sum_{l=0}^{L-1} Layer Shift^(l)
    """
    S = math.sqrt(d_h / r_target)
    measured_shifts = []
    
    for seq_len in seq_lengths:
        for name, _, _ in models_to_test:
            dir_name = f"induction_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}"
            ckpt_path = f"data/induction_degradation_models/seqlen_{seq_len}/{dir_name}/checkpoint.pt"
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                sd = ckpt.get('model_state_dict', ckpt)
                
                total_shift = 0.0
                for l in range(num_layers):
                    wq_k = f"blocks.{l}.attn.W_q.weight"
                    wk_k = f"blocks.{l}.attn.W_k.weight"
                    if wq_k in sd and wk_k in sd:
                        W_q = sd[wq_k].view(4, d_h, 128)[:, :r_target, :]
                        W_k = sd[wk_k].view(4, d_h, 128)[:, :r_target, :]
                        prod_per_chan = W_q.norm(p=2, dim=-1) * W_k.norm(p=2, dim=-1)
                        layer_shift = S * (1.0 / math.sqrt(d_h)) * prod_per_chan.sum(dim=-1).mean().item()
                        total_shift += layer_shift
                        
                if total_shift > 0:
                    measured_shifts.append(total_shift)
                    
    mean_shift = np.mean(measured_shifts) if len(measured_shifts) > 0 else None
    return mean_shift

def fit_theory(N_vals, acc_vals, r=10, N_range=None, measured_shift=None):
    N_arr = np.array(N_vals, dtype=float)
    acc_arr = np.array(acc_vals, dtype=float)
    
    # Throw away un-converged points (< 5% accuracy) to fit Section 3.11 capacity degradation on grokked points
    mask = acc_arr >= 5.0
    N_fit = N_arr[mask]
    acc_fit = acc_arr[mask]
    
    if len(N_fit) < 2:
        print("Not enough valid data points (acc >= 5%) to fit theoretical curve.")
        return None, None, None, None, None, None
        
    def phi(x):
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

    # Pure Section 3.11 theoretical model: P = 100 * Phi(mu - b_n)
    def model_fn(N, gamma, c):
        n = np.maximum(2.0, c * N)
        log_n = np.log(n)
        b_n = np.sqrt(2.0 * log_n) - (np.log(log_n) + np.log(4.0 * np.pi)) / (2.0 * np.sqrt(2.0 * log_n))
        mu = np.sqrt(gamma * r / (N - r))
        return 100.0 * phi(mu - b_n)
        
    try:
        popt, _ = curve_fit(model_fn, N_fit, acc_fit, p0=[1500.0, 0.001], bounds=([0.0, 1e-6], [np.inf, 0.5]))
        gamma_fit, c_fit = popt[0], popt[1]
        
        preds_fit = model_fn(N_fit, gamma_fit, c_fit)
        
        # Calculate error metrics strictly on grokked points
        mae = np.mean(np.abs(preds_fit - acc_fit))
        rmse = np.sqrt(np.mean((preds_fit - acc_fit) ** 2))
        
        ss_res = np.sum((acc_fit - preds_fit) ** 2)
        ss_tot = np.sum((acc_fit - np.mean(acc_fit)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        
        # Dense points for theoretical curve extending out of frame
        if N_range is not None:
            min_x, max_x = N_range
        else:
            min_x, max_x = min(N_fit), max(N_fit)
            
        x_span = max_x - min_x
        N_dense = np.linspace(max(10.0, min_x - 0.05 * x_span), max_x + 0.05 * x_span, 400)
        acc_dense = model_fn(N_dense, gamma_fit, c_fit)
        
        print(f"\n========================================")
        print(f" THEORY FIT RESULTS (GROKKED POINTS, r={r})")
        print(f"========================================")
        print(f" Points fitted (acc >= 5%): {len(N_fit)}")
        print(f" Sequence Lengths: {N_fit.astype(int).tolist()}")
        print(f" Empirical Accuracies: {[round(a, 2) for a in acc_fit.tolist()]}")
        print(f" Predicted Accuracies: {[round(a, 2) for a in preds_fit.tolist()]}")
        print(f" Fitted Signal Gain (γ_fit): {gamma_fit:.4f}")
        if measured_shift is not None:
            print(f" Measured Outlier Signal Shift (μ_weights): {measured_shift:.4f} (Accumulated across 4 layers)")
        print(f" Fitted Distractor Ratio (c): {c_fit:.6f}")
        print(f" Mean Absolute Error (MAE): {mae:.3f}%")
        print(f" Root Mean Square Error (RMSE): {rmse:.3f}%")
        print(f" Theoretical R^2 Score: {r2:.4f}")
        print(f"========================================\n")
        
        return gamma_fit, c_fit, r2, mae, N_dense, acc_dense
    except Exception as e:
        print(f"Error during theoretical curve fitting: {e}")
        return None, None, None, None, None, None

def plot_induction_degradation(config=None):
    if config is None:
        config = InductionDegradationExperimentConfig()
        
    models_to_test = config.models_to_test
    seq_lengths = sorted(config.seq_lengths)
    
    results = {name: [] for name, _, _ in models_to_test}
    valid_seq_lens = {name: [] for name, _, _ in models_to_test}
    
    for seq_len in seq_lengths:
        for name, attn_type, r_val in models_to_test:
            dir_name = f"induction_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}"
            ckpt_path = f"data/induction_degradation_models/seqlen_{seq_len}/{dir_name}/checkpoint.pt"
            
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location='cpu')
                history = ckpt.get('metadata', {}).get('history', {})
                accs = history.get('acc', [])
                final_acc = max(accs) if len(accs) > 0 else 0.0
            else:
                final_acc = 0.0
                
            results[name].append(final_acc)
            valid_seq_lens[name].append(seq_len)

    # Save to data/plots/induction/
    output_dir = "data/plots/induction"
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    plt.figure(figsize=(10, 6))
    
    has_data = False
    in_frame_ticks = set()
    y_min_limit = 84.5
    y_max_limit = 100.5
    
    # Extract measured empirical signal shift
    measured_shift = extract_measured_gamma(seq_lengths, models_to_test, r_target=10)
    
    for name, attn_type, r_val in models_to_test:
        if len(results[name]) > 0:
            N_vals = np.array(valid_seq_lens[name])
            acc_vals = np.array(results[name])
            
            # Filter in-frame points (acc >= y_min_limit) for X-axis ticks
            in_frame_mask = acc_vals >= y_min_limit
            in_frame_seqs = N_vals[in_frame_mask]
            for s in in_frame_seqs:
                in_frame_ticks.add(int(s))
            
            # Plot empirical data
            plt.plot(N_vals, acc_vals, label=f"Empirical {name}", marker='o', markersize=6, linewidth=2)
            has_data = True
            
            # Fit theory curve over full sequence length range (filtering acc < 5% inside fit_theory)
            r_target = r_val if r_val is not None else 10
            x_range = (min(in_frame_seqs), max(in_frame_seqs)) if len(in_frame_seqs) > 0 else (min(N_vals), max(N_vals))
            gamma_fit, c_fit, r2, mae, N_dense, acc_dense = fit_theory(N_vals, acc_vals, r=r_target, N_range=x_range, measured_shift=measured_shift)
            if mae is not None:
                plt.plot(N_dense, acc_dense, '--', color='tab:red', linewidth=2, label=f"Theory Fit (MAE={mae:.1f}%)")
            
    if not has_data:
        print("No degradation checkpoints found to plot.")
        plt.close()
        return

    plt.title("Context Length Degradation (HOFA r=10)", fontsize=13, pad=12)
    plt.xlabel("Sequence Length (N)", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    
    sorted_ticks = sorted(list(in_frame_ticks)) if len(in_frame_ticks) > 0 else sorted(seq_lengths)
    plt.xticks(sorted_ticks, sorted_ticks, rotation=45, ha='right')
    
    if len(sorted_ticks) > 1:
        x_pad = (sorted_ticks[-1] - sorted_ticks[0]) * 0.05
        plt.xlim(sorted_ticks[0] - x_pad, sorted_ticks[-1] + x_pad)
        
    plt.ylim(y_min_limit, y_max_limit)
    plt.yticks([85, 90, 95, 100], ['85', '90', '95', '100'])
    plt.legend(loc='best', frameon=True)
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, "r10_context_degradation.pdf")
    plt.savefig(plot_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.close()
    
    print(f"\nDegradation plot successfully saved to {plot_path}")
