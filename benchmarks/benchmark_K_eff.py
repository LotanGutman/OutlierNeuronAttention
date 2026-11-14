import os
import sys
import argparse
import pickle
from typing import Dict, Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from benchmarks.benchmarks_configs import KEffExperimentConfig, CACHE_PATH


def get_eval_sequences(tokenizer, config: KEffExperimentConfig) -> torch.Tensor:
    """
    Stream FineWeb-Edu and collect enough tokens to form `num_sequences` of `seq_len`.
    """
    dataset = load_dataset(
        config.dataset_name,
        name=config.dataset_config,
        split=config.dataset_split,
        streaming=True
    )
    
    collected_tokens = []
    sequences = []
    
    for item in dataset:
        tokens = tokenizer.encode(item["text"])
        collected_tokens.extend(tokens)
        
        while len(collected_tokens) >= config.seq_len:
            sequences.append(collected_tokens[:config.seq_len])
            collected_tokens = collected_tokens[config.seq_len:]
            
            if len(sequences) == config.num_sequences:
                return torch.tensor(sequences, dtype=torch.long)
    return torch.tensor(sequences, dtype=torch.long)


def measure_model_channel_power(
    model_name: str,
    config: KEffExperimentConfig,
    device: str = "cuda"
) -> Dict[str, Any]:
    print(f"\n{'='*70}\nEvaluating: {model_name}:")
    
    hf_token = os.environ.get("HF_TOKEN", None)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
        token=hf_token,
        device_map="auto"
    )
    model.eval()
    
    # 1. Collect evaluation sequences
    print(f"Collecting {config.num_sequences} sequences of length {config.seq_len} from FineWeb-Edu...")
    input_ids = get_eval_sequences(tokenizer, config)
    target_device = next(model.parameters()).device
    input_ids = input_ids.to(target_device)
    
    # 2. Setup hook depending on architecture
    captured = []
    hooked_module = None
    orig_attn_fn = None
    
    model_type = getattr(model.config, "model_type", "").lower()
    if model_type == "gpt_neox":
        import transformers.models.gpt_neox.modeling_gpt_neox as mod
    elif model_type in ("llama", "meta-llama"):
        import transformers.models.llama.modeling_llama as mod
    elif model_type in ("qwen2", "qwen"):
        import transformers.models.qwen2.modeling_qwen2 as mod
    elif model_type == "gpt2":
        import transformers.models.gpt2.modeling_gpt2 as mod
    else:
        # Generic fallback
        try:
            import transformers.models.llama.modeling_llama as mod
        except Exception:
            mod = None

    if mod is not None and hasattr(mod, "eager_attention_forward"):
        hooked_module = mod
        orig_attn_fn = mod.eager_attention_forward
        def hook(m, q, k, v, attention_mask=None, **kw):
            captured.append((q.detach(), k.detach()))
            return orig_attn_fn(m, q, k, v, attention_mask=attention_mask, **kw)
        mod.eager_attention_forward = hook
    else:
        raise RuntimeError(f"Unsupported model type {model_type} for eager attention interception")

    eval_start = config.eval_start_token
    all_C = []
    all_Neff = []

    try:
        with torch.no_grad():
            for s_idx in tqdm(range(config.num_sequences), desc="Sequences"):
                captured.clear()
                seq = input_ids[s_idx:s_idx+1]
                model(seq)
                
                L = len(captured)
                H_q = captured[0][0].shape[1]
                H_k = captured[0][1].shape[1]
                d_h = captured[0][0].shape[-1]
                
                seq_C = np.zeros((L, H_q, d_h))
                seq_N = np.zeros((L, H_q))
                
                for l_idx in range(L):
                    Q, K = captured[l_idx]
                    # Q: [1, H_q, N, d_h], K: [1, H_k, N, d_h]
                    Q_eval = Q[0, :, eval_start:, :].float() # [H_q, N_eval, d_h]
                    K_eval = K[0].float()                   # [H_k, N_full, d_h]
                    
                    # Handle GQA (repeat K heads to match Q heads if needed)
                    if H_q != H_k:
                        K_eval = torch.repeat_interleave(K_eval, H_q // H_k, dim=0)
                        
                    # Causal cumulative sum for key squared norms
                    k_cumsum = torch.cumsum(K_eval ** 2, dim=1)[:, eval_start:, :] # [H_q, N_eval, d_h]
                    
                    # E_c = sum_{i, j<=i} S_c(i, j)^2 = sum_i Q_{i, c}^2 * sum_{j<=i} K_{j, c}^2 / d_h
                    E_c = ((Q_eval ** 2) * k_cumsum).sum(dim=1) / d_h # [H_q, d_h]
                    sum_E = E_c.sum(dim=-1, keepdim=True) + 1e-12
                    p_c = E_c / sum_E # [H_q, d_h]
                    
                    # 1. Top-r cumulative power
                    p_sorted, _ = torch.sort(p_c, descending=True, dim=-1)
                    seq_C[l_idx] = torch.cumsum(p_sorted, dim=-1).cpu().numpy()
                    
                    # 2. Shannon entropic effective channels
                    entropy = - (p_c * torch.log(p_c + 1e-12)).sum(dim=-1) # [H_q]
                    seq_N[l_idx] = torch.exp(entropy).cpu().numpy()
                    
                all_C.append(seq_C)
                all_Neff.append(seq_N)

    finally:
        if hooked_module and orig_attn_fn:
            hooked_module.eager_attention_forward = orig_attn_fn

    all_C = np.array(all_C)       # [N_SEQ, L, H_q, d_h]
    all_Neff = np.array(all_Neff) # [N_SEQ, L, H_q]
    
    # Layer-wise aggregates
    C_matrix = all_C.mean(axis=(0, 2)).T # [d_h, L]
    Neff_means = all_Neff.mean(axis=(0, 2)) # [L]
    Neff_p25 = np.percentile(all_Neff.reshape(config.num_sequences, L, -1), 25, axis=(0, 2))
    Neff_p75 = np.percentile(all_Neff.reshape(config.num_sequences, L, -1), 75, axis=(0, 2))
    
    overall_C_at_16 = float(C_matrix[min(15, d_h - 1), :].mean())
    print(f"Model {model_name} Complete: Overall Mean C(r=16) = {overall_C_at_16 * 100:.2f}%")

    summary = {
        "model_name": model_name,
        "num_layers": L,
        "num_heads": H_q,
        "head_dim": d_h,
        "C_matrix": C_matrix,
        "N_eff_means": Neff_means,
        "N_eff_p25": Neff_p25,
        "N_eff_p75": Neff_p75,
        "overall_C_at_16": overall_C_at_16
    }
    
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return summary


def run_k_eff_experiment(models_override=None, plot_after=False):
    config = KEffExperimentConfig()
    if isinstance(models_override, KEffExperimentConfig):
        config = models_override
        models_to_run = config.models_to_test
    elif models_override:
        models_to_run = models_override
    else:
        models_to_run = config.models_to_test
    cache_path = os.path.join(CACHE_PATH, config.cache_file_name)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    results = {}
    if os.path.exists(cache_path):
        print(f"Loading existing cache from {cache_path}")
        with open(cache_path, "rb") as f:
            results = pickle.load(f)
            
    for m in models_to_run:
        print(f"\nProcessing {m}...")
        try:
            m_res = measure_model_channel_power(m, config)
            results[m] = m_res
            with open(cache_path, "wb") as f:
                pickle.dump(results, f)
            print(f"Saved checkpoint to {cache_path}")
        except Exception as e:
            print(f"Error evaluating {m}: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"\nAll requested models processed. Cache saved at {cache_path}")
    
    if plot_after:
        from benchmarks.plotting.plot_K_eff import plot_heatmap_cumsum_all
        plot_heatmap_cumsum_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-Channel QK-Power Proxy & Entropic d_eff Analysis")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names to run")
    parser.add_argument("--plot", action="store_true", help="Generate plots after measurement")
    args = parser.parse_args()
    
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    run_k_eff_experiment(models_override=models, plot_after=args.plot)
