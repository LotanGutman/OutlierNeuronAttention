import torch
import os
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

def fit_theory(N_vals, acc_vals, r=10, N_range=None):
    N_arr = np.array(N_vals, dtype=float)
    acc_arr = np.array(acc_vals, dtype=float)
    
    # Throw away points with less than 5% accuracy (past threshold of normal growth)
    mask = acc_arr >= 5.0
    N_fit = N_arr[mask]
    acc_fit = acc_arr[mask]
    
    if len(N_fit) < 2:
        print("Not enough valid data points (accuracy >= 5%) to fit theoretical curve.")
        return None, None, None, None
        
    def phi(x):
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

    def model_fn(N, gamma):
        n = np.maximum(2.0, N / 2.0 - 1.0)
        log_n = np.log(n)
        b_n = np.sqrt(2.0 * log_n) - (np.log(log_n) + np.log(4.0 * np.pi)) / (2.0 * np.sqrt(2.0 * log_n))
        mu = np.sqrt(gamma * r / (N - r))
        return 100.0 * phi(mu - b_n)
        
    try:
        popt, _ = curve_fit(model_fn, N_fit, acc_fit, p0=[500.0], bounds=(0.0, np.inf))
        gamma_fit = popt[0]
        
        preds_fit = model_fn(N_fit, gamma_fit)
        ss_res = np.sum((acc_fit - preds_fit) ** 2)
        ss_tot = np.sum((acc_fit - np.mean(acc_fit)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        
        # Dense points for theoretical curve extending out of frame on both sides
        if N_range is not None:
            min_x, max_x = N_range
        else:
            min_x, max_x = min(N_fit), max(N_fit)
            
        x_span = max_x - min_x
        N_dense = np.linspace(max(10.0, min_x - 0.15 * x_span), max_x + 0.15 * x_span, 300)
        acc_dense = model_fn(N_dense, gamma_fit)
        
        print(f"\n========================================")
        print(f" THEORY FIT RESULTS (r={r})")
        print(f"========================================")
        print(f" Points fitted (acc >= 5%): {len(N_fit)}")
        print(f" Sequence Lengths: {N_fit.astype(int).tolist()}")
        print(f" Fitted Projection Gamma (γ): {gamma_fit:.4f}")
        print(f" Theoretical R^2 Score: {r2:.4f}")
        print(f"========================================\n")
        
        return gamma_fit, r2, N_dense, acc_dense
    except Exception as e:
        print(f"Error during theoretical curve fitting: {e}")
        return None, None, None, None

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
                if len(accs) > 0:
                    final_acc = max(accs)
                    results[name].append(final_acc)
                    valid_seq_lens[name].append(seq_len)

    os.makedirs("data/plots/induction", exist_ok=True)
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    plt.figure(figsize=(10, 6))
    
    has_data = False
    in_frame_ticks = set()
    y_min_limit = 84.5
    y_max_limit = 100.5
    
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
            
            # Fit theory curve and extend it out of frame in both X directions
            r_target = r_val if r_val is not None else 10
            x_range = (min(in_frame_seqs), max(in_frame_seqs)) if len(in_frame_seqs) > 0 else (min(N_vals), max(N_vals))
            gamma_fit, r2, N_dense, acc_dense = fit_theory(N_vals, acc_vals, r=r_target, N_range=x_range)
            if r2 is not None:
                plt.plot(N_dense, acc_dense, '--', color='tab:red', linewidth=2, label=f"Theory Fit ($R^2={r2:.3f}$)")
            
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
    
    plot_path = "data/plots/induction/r10_context_degradation.pdf"
    plt.savefig(plot_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.close()
    
    print(f"\nDegradation plot successfully saved to {plot_path}")
