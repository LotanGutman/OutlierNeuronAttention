import sys
try:
    import lzma
except ImportError:
    import types
    dummy_lzma = types.ModuleType('lzma')
    sys.modules['lzma'] = dummy_lzma

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

os.environ["HF_TOKEN"] = "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"

from benchmarks.benchmarks_configs import KEffExperimentConfig, CACHE_PATH

def get_eval_sequences(tokenizer, config: KEffExperimentConfig):
    """
    Stream FineWeb-Edu and collect enough tokens to form `num_sequences` of `seq_len`.
    """
    dataset = load_dataset(config.dataset_name, name=config.dataset_config, split=config.dataset_split, streaming=True)
    
    collected_tokens = []
    sequences = []
    
    for item in dataset:
        tokens = tokenizer.encode(item["text"])
        collected_tokens.extend(tokens)
        
        while len(collected_tokens) >= config.seq_len:
            sequences.append(collected_tokens[:config.seq_len])
            collected_tokens = collected_tokens[config.seq_len:]
            
            if len(sequences) == config.num_sequences:
                return torch.tensor(sequences)

def measure_k_eff(model_name, config: KEffExperimentConfig):
    print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, output_attentions=True, token=os.environ["HF_TOKEN"], device_map="auto")
    model.eval()
    
    print("Collecting validation sequences from FineWeb-Edu...")
    input_ids = get_eval_sequences(tokenizer, config)
    target_device = next(model.parameters()).device
    input_ids = input_ids.to(target_device)
    
    batch_size = config.batch_size
    all_cumsums = {}
    query_counts = {}
    
    print("Running forward passes and computing cumulative attention mass...")
    with torch.no_grad():
        for i in tqdm(range(0, config.num_sequences, batch_size)):
            batch = input_ids[i:i+batch_size]
            
            outputs = model(batch, output_attentions=True)
            attentions = outputs.attentions
            
            for layer_idx, layer_attn in enumerate(attentions):
                if layer_idx not in all_cumsums:
                    all_cumsums[layer_idx] = torch.zeros(config.seq_len, device=layer_attn.device, dtype=torch.float64)
                    query_counts[layer_idx] = 0
                    
                # Grab the attention distributions for the second half of the sequence: tokens 512 to 1024
                # layer_attn shape: [batch_size, num_heads, seq_len, seq_len]
                eval_attn = layer_attn[:, :, 512:, :].float()
                
                # Sort and cumulative sum to get mass captured by top-k tokens
                sorted_attn, _ = torch.sort(eval_attn, descending=True, dim=-1)
                cumsum_attn = torch.cumsum(sorted_attn, dim=-1)
                
                all_cumsums[layer_idx] += cumsum_attn.sum(dim=(0, 1, 2))
                query_counts[layer_idx] += eval_attn.shape[0] * eval_attn.shape[1] * eval_attn.shape[2]
                
    # Average across all queries
    for layer_idx in all_cumsums:
        all_cumsums[layer_idx] = (all_cumsums[layer_idx] / query_counts[layer_idx]).cpu().numpy()
    
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return all_cumsums

import pickle

def run_k_eff_experiment():
    config = KEffExperimentConfig()
    
    results = {}
    
    cache_path = os.path.join(CACHE_PATH, config.cache_file_name)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if os.path.exists(cache_path):
        print(f"Loading cached results from {cache_path}")
        with open(cache_path, "rb") as f:
            results = pickle.load(f)
            
    for m in config.models_to_test:
        if m in results:
            print(f"Skipping {m} (already cached)")
            continue
        layerwise_cumsum = measure_k_eff(m, config)
        results[m] = layerwise_cumsum
        
        # Save incrementally
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
        
    print(f"\nSaved layerwise cumulative mass results to {cache_path}.")

if __name__ == "__main__":
    run_k_eff_experiment()
