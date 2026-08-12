import os
import torch
import pickle
import math
import torch.nn.functional as F
from tqdm import tqdm

from training.training_config import make_125M_HOFA, make_125M_MHA
from benchmarks.benchmarks_configs import DistanceExperimentConfig
from training.data_utils import FastTokenLoader
from src.HybridOutlierFactorizedAttentionTrain import SubwordLM
from src.modules.checkpointing import load_checkpoint
import src.HybridOutlierFactorizedAttentionTrain as HOFA_Train
# pyrefly: ignore [missing-import]
from benchmarks.plottin.plot_distance import run_plot_distance

# Global storage for hooked variables
captured_vars = {}

# We will monkey-patch the sdpa kernel to capture exact attention probabilities
original_sdpa = F.scaled_dot_product_attention

def hooked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))
    
    L, S = query.size(-2), key.size(-2)
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(~temp_mask, float("-inf"))
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(~attn_mask, float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    # Save the exact weights
    if 'attn_weights' not in captured_vars:
        captured_vars['attn_weights'] = []
    captured_vars['attn_weights'].append(attn_weight.detach())
    
    return attn_weight @ value

# Monkey-patch the forward of HOFA to capture Q, K, and gate_logits
original_hofa_forward = HOFA_Train.HybridOutlierFactorizedAttention.forward

def hooked_hofa_forward(self, x):
    B, N, D = x.shape
    scale_factor = self.d_head ** 0.25

    Q = (self.W_q(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
    K = (self.W_k(x) / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
    V = self.W_v(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)

    gate_logits, mix_g = self._compute_gates_optimized(Q, K)
    
    if 'gate_logits' not in captured_vars:
        captured_vars['gate_logits'] = []
    if 'Q' not in captured_vars:
        captured_vars['Q'] = []
    if 'K' not in captured_vars:
        captured_vars['K'] = []
        
    captured_vars['gate_logits'].append(gate_logits.detach())
    captured_vars['Q'].append(Q.detach())
    captured_vars['K'].append(K.detach())
    
    return original_hofa_forward(self, x)

def compute_distances(model_name: str, config: DistanceExperimentConfig):
    # Determine the model config factory
    if model_name == "125M_MHA":
        model_config_wrap = make_125M_mha()
    elif model_name == "125M_HOFA":
        model_config_wrap = make_125M_hofa()
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    device = config.device
    model_cfg = model_config_wrap.model_config
    model = SubwordLM(model_config_wrap.vocab_size, model_cfg)
    model.to(device)
    
    checkpoint_dir = f"data/training/{model_config_wrap.model_name}"
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    try:
        load_checkpoint(model, optimizer, model_config_wrap.model_name, -1, checkpoint_dir)
        print(f"Loaded checkpoint for {model_name}")
    except Exception as e:
        print(f"Failed to load checkpoint for {model_name}. Skipping. ({e})")
        return None
        
    model.eval()
    
    # Load dataset
    val_cache_path = f"data/datasets/data_{model_name.split('_')[0]}_val_cache.bin"
    loader = FastTokenLoader(val_cache_path, config.batch_size, config.seq_len)
    
    total_seqs_run = 0
    max_seqs = config.num_sequences
    
    # For accumulation: (num_layers, num_heads) -> scalars
    num_layers = model_cfg.num_layers
    num_heads = model_cfg.num_heads
    
    total_outlier_dist = torch.zeros(num_layers, num_heads, device=device)
    total_inlier_dist = torch.zeros(num_layers, num_heads, device=device)
    
    batches_processed = 0
    
    with torch.no_grad():
        pbar = tqdm(total=max_seqs, desc=f"Evaluating {model_name}")
        while total_seqs_run < max_seqs:
            x, _, _ = loader.get_batch()
            x = x.to(device)
            current_batch_size = x.size(0)
            
            if total_seqs_run + current_batch_size > max_seqs:
                # Truncate batch if we exceed max_seqs
                current_batch_size = max_seqs - total_seqs_run
                x = x[:current_batch_size]
                
            total_seqs_run += current_batch_size
            pbar.update(current_batch_size)
            
            captured_vars.clear()
            
            # Forward pass to trigger hooks
            model(x)
            
            # Now compute distances from captured variables
            # captured_vars['attn_weights'] has length = num_layers
            # Shape of each: (B, H, N, N)
            
            # i_minus_j matrix (N, N)
            N = x.size(1)
            i_idx = torch.arange(N, device=device).unsqueeze(1)
            j_idx = torch.arange(N, device=device).unsqueeze(0)
            dist_mat = (i_idx - j_idx).float()
            # Only consider j <= i
            mask_le = (j_idx <= i_idx)
            dist_mat = dist_mat * mask_le
            
            for layer_idx in range(num_layers):
                # exact/outlier distance calculation
                p_ij = captured_vars['attn_weights'][layer_idx] # (B, H, N, N)
                
                # Drop first 5 tokens (attention sinks)
                p_ij = p_ij[:, :, 5:, :]
                dist_mat_crop = dist_mat[5:, :]
                
                # sum_j p_ij * (i-j) -> shape (B, H, N-5)
                # p_ij is already normalized so sum_j p_ij = 1
                dist_outlier = torch.sum(p_ij * dist_mat_crop.unsqueeze(0).unsqueeze(0), dim=-1)
                
                # Average over sequence and batch -> (H,)
                avg_dist_outlier = dist_outlier.mean(dim=(0, 2))
                total_outlier_dist[layer_idx] += avg_dist_outlier
                
                # GLA inlier calculation (if it's a HOFA model with GLA)
                if model_name == "125M_HOFA" and model_cfg.r[layer_idx] < model_cfg.d_head:
                    Q = captured_vars['Q'][layer_idx] # (B, H, N, d_head)
                    K = captured_vars['K'][layer_idx]
                    gate_logits = captured_vars['gate_logits'][layer_idx].squeeze(-1) # (B, H, N)
                    
                    # Log-gamma
                    log_gamma = F.logsigmoid(-gate_logits) # (B, H, N)
                    S = torch.cumsum(log_gamma, dim=2) # (B, H, N)
                    
                    # We want D_ij = exp(S_{i-1} - S_j) for j < i
                    # Let's build a dense N x N matrix for S_{i-1} and S_j
                    S_i_minus_1 = torch.cat([torch.zeros_like(S[:, :, :1]), S[:, :, :-1]], dim=2)
                    
                    diff = S_i_minus_1.unsqueeze(3) - S.unsqueeze(2)
                    
                    # Enforce j < i mask!
                    mask_lt = (j_idx < i_idx)
                    diff = diff.masked_fill(~mask_lt.unsqueeze(0).unsqueeze(0), -float('inf'))
                    
                    # D_ij shape: (B, H, N, N)
                    D_ij = torch.exp(diff)
                    
                    # Content matrix A_ij = Q_i * K_j
                    # K needs to be split to just the GLA portion!
                    r = model_cfg.r[layer_idx]
                    q_gla = Q[:, :, :, r:].to(torch.float32)
                    k_gla = K[:, :, :, r:].to(torch.float32)
                    
                    A_ij = torch.matmul(q_gla, k_gla.transpose(-2, -1)) # (B, H, N, N)
                    
                    # Pseudo-probs (since D_ij is 0 where j >= i, C_ij will be 0 there too)
                    C_ij = D_ij * A_ij
                    
                    # Normalization denominator sum_{k < i} |C_{ik}|
                    C_ij_abs = torch.abs(C_ij)
                    denom = torch.sum(C_ij_abs, dim=-1, keepdim=True)
                    denom = torch.clamp(denom, min=1e-9)
                    
                    P_ij = C_ij_abs / denom
                    
                    # Drop first 5 tokens
                    P_ij = P_ij[:, :, 5:, :]
                    
                    dist_inlier = torch.sum(P_ij * dist_mat_crop.unsqueeze(0).unsqueeze(0), dim=-1)
                    avg_dist_inlier = dist_inlier.mean(dim=(0, 2))
                    total_inlier_dist[layer_idx] += avg_dist_inlier
                    
            batches_processed += 1
            
    # Average over all batches
    total_outlier_dist /= batches_processed
    if model_name == "125M_HOFA":
        total_inlier_dist /= batches_processed
        
    return {
        "outlier": total_outlier_dist.cpu(), # (num_layers, num_heads)
        "inlier": total_inlier_dist.cpu() if model_name == "125M_HOFA" else None
    }

def run_distance_experiment():
    print("Monkey-patching F.scaled_dot_product_attention and HOFA.forward...")
    F.scaled_dot_product_attention = hooked_sdpa
    HOFA_Train.HybridOutlierFactorizedAttention.forward = hooked_hofa_forward

    config = DistanceExperimentConfig()
    results = {}
    
    cache_path = os.path.join("data/experiments_cache", config.cache_file_name)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if os.path.exists(cache_path):
        print(f"Loading cached results from {cache_path}")
        with open(cache_path, "rb") as f:
            results = pickle.load(f)
            
    for m in config.models_to_test:
        if m in results:
            print(f"Skipping {m} (already cached)")
            continue
            
        distances = compute_distances(m, config)
        if distances is not None:
            results[m] = distances
            with open(cache_path, "wb") as f:
                pickle.dump(results, f)
            
    print(f"\nSaved effective distance results to {cache_path}.")

    # Restore original functions
    F.scaled_dot_product_attention = original_sdpa
    HOFA_Train.HybridOutlierFactorizedAttention.forward = original_hofa_forward

    run_plot_distance()

if __name__ == "__main__":
    run_distance_experiment()
