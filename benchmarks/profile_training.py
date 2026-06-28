import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.HybridOutlierFactorizedAttention import HybridOutlierFactorizedAttention
from src.config import ModelConfig
from benchmarks.benchmarks_configs import CACHE_PATH

class StandardMHAWrapper(nn.Module):
    def __init__(self, d_head, num_heads):
        super().__init__()
        self.d_head = d_head
        self.num_heads = num_heads
        self.W_q = nn.Identity()
        self.W_k = nn.Identity()
        self.W_v = nn.Identity()
        self.out_proj = nn.Identity()
        
    def forward(self, x):
        B, N, D = x.shape
        scale_factor = self.d_head ** 0.25
        Q = (x / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        K = (x / scale_factor).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = x.view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=1.0)
        Y = Y.transpose(1, 2).reshape(B, N, D)
        return Y

def profile_fwd_bwd(model, x, warmup=3, active=5):
    for _ in range(warmup):
        if x.grad is not None:
            x.grad.zero_()
        out = model(x)
        loss = out.sum()
        loss.backward(retain_graph=True)
            
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(active):
        if x.grad is not None:
            x.grad.zero_()
        out = model(x)
        loss = out.sum()
        loss.backward(retain_graph=True)
            
    end_event.record()
    torch.cuda.synchronize()
    
    latency = start_event.elapsed_time(end_event) / active
    memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return latency, memory

def main():
    device = "cuda"
    seq_lengths = [32768, 65536, 131072]
    
    model_cfg = ModelConfig(d_model=256, num_heads=4, r=16, chunk_size=64)
    d_head = model_cfg.d_head
    num_heads = model_cfg.num_heads
    
    hofa = HybridOutlierFactorizedAttention(model_cfg).to(device).to(torch.bfloat16)
    hofa.W_q = hofa.W_k = hofa.W_v = hofa.out_proj = nn.Identity()
    hofa.train()
    
    mha = StandardMHAWrapper(d_head, num_heads).to(device).to(torch.bfloat16)
    mha.train()
    
    cache_path = os.path.join(CACHE_PATH, "profile_training_cache.pt")
    force_rerun = True
    
    if os.path.exists(cache_path) and not force_rerun:
        print(f"Loading cached results from {cache_path}")
        results = torch.load(cache_path)
    else:
        results = []
        
        print("=" * 80)
        print(f"Training Profiling (Fwd+Bwd) (d_model={model_cfg.d_model}, num_heads={num_heads}, r={model_cfg.r})")
        print("=" * 80)
        print(f"{'Seq Len':<10} | {'Model':<6} | {'Latency (ms)':<15} | {'Max VRAM (MB)':<15}")
        print("-" * 80)
        
        for seq_len in seq_lengths:
            x = torch.randn(1, seq_len, model_cfg.d_model, device=device, dtype=torch.bfloat16).requires_grad_(True)
            
            # Profile MHA
            try:
                lat_mha, mem_mha = profile_fwd_bwd(mha, x)
                print(f"{seq_len:<10} | {'MHA':<6} | {lat_mha:<15.2f} | {mem_mha:<15.2f}")
            except Exception as e:
                lat_mha, mem_mha = None, None
                print(f"{seq_len:<10} | {'MHA':<6} | {'FAILED/OOM':<15} | {'-':<15}")
                
            # Profile HOFA
            try:
                lat_hofa, mem_hofa = profile_fwd_bwd(hofa, x)
                print(f"{seq_len:<10} | {'HOFA':<6} | {lat_hofa:<15.2f} | {mem_hofa:<15.2f}")
            except Exception as e:
                lat_hofa, mem_hofa = None, None
                print(f"{seq_len:<10} | {'HOFA':<6} | {'FAILED/OOM':<15} | {'-':<15}")
                
            print("-" * 80)
            results.append((seq_len, lat_mha, mem_mha, lat_hofa, mem_hofa))
            
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(results, cache_path)
        print(f"Results saved to {cache_path}")

if __name__ == "__main__":
    main()
