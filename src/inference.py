import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os
from src.HybridOutlierFactorizedAttention import SubwordLM

DEBUG_MODE = False  # Set to True to enable debug prints

class InferenceEngine:
    def __init__(self, experiment_cfg, checkpoint_path: str = None):
        self.model_cfg = experiment_cfg.model_config
        self.device = torch.device(experiment_cfg.device)
        self.model_name = experiment_cfg.model_name
        self.tokenizer = tiktoken.get_encoding(self.model_cfg.tokenizer_name)
        self.vocab_size = self.tokenizer.n_vocab

        self.model = SubwordLM(self.vocab_size, self.model_cfg).to(self.device)
        self.model.eval()

        if checkpoint_path is None:
            checkpoint_dir = f"data/training/{self.model_name}"
            checkpoint_path = self._find_latest_checkpoint(checkpoint_dir)
        else:
            checkpoint_path = self._resolve_path(checkpoint_path)
            
        self._load_checkpoint(checkpoint_path)

    def _resolve_path(self, path):
        """Resolves Windows-style paths to WSL paths if running on Linux/WSL."""
        if os.path.exists(path):
            return path
        
        # Fallback for WSL: convert C:\Users\... to /mnt/c/Users/...
        if os.name == 'posix' and len(path) > 2 and path[1] == ':':
            drive = path[0].lower()
            relative_path = path[3:].replace('\\', '/')
            wsl_path = f"/mnt/{drive}/{relative_path}"
            if os.path.exists(wsl_path):
                print(f"Path Fallback (WSL): {path} -> {wsl_path}")
                return wsl_path
        return path

    def _find_latest_checkpoint(self, dir_path):
        dir_path = self._resolve_path(dir_path)
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Checkpoint directory {dir_path} does not exist.")
        
        files = [f for f in os.listdir(dir_path) if f.endswith('.pt')]
        if not files:
            raise FileNotFoundError(f"No checkpoints found in {dir_path}.")
        
        steps = []
        for f in files:
            if 'hybrid_attn_step' in f:
                try:
                    step = int(f.split('_')[-1].split('.')[0])
                    steps.append(step)
                except ValueError:
                    continue
        
        if not steps:
            raise FileNotFoundError("No valid hybrid_attn_step_*.pt files found.")
            
        latest = max(steps)
        return os.path.join(dir_path, f"hybrid_attn_step_{latest}.pt")

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt['model_state_dict']
        
        for key, tensor in list(state.items()):
            if '_cached_outlier_idx' in key or '_cached_inlier_idx' in key:
                *path_parts, attr = key.split('.')
                obj = self.model
                for p in path_parts:
                    obj = getattr(obj, p)
                if getattr(obj, attr) is None:
                    setattr(obj, attr, torch.zeros_like(tensor))
        
        missing_keys, unexpected_keys = self.model.load_state_dict(state, strict=False)


        if DEBUG_MODE:
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
        
        if any('out_proj' in k for k in missing_keys):
            print("--- Initializing missing out_proj weights as Identity ---")
            with torch.no_grad():
                for name, module in self.model.named_modules():
                    if 'out_proj' in name and isinstance(module, nn.Linear):
                        nn.init.eye_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

        print(f"Loaded checkpoint from {path}")
    
    def generate(self, prompt: str = "", max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, repetition_penalty = 1.15, stream: bool = False):
        """Generate text from a prompt. If prompt is empty, start from eot token."""
        self.model.eval()
        if prompt:
            ids = self.tokenizer.encode(prompt)
        else:
            ids = [self.tokenizer.eot_token]
            
        context = torch.tensor([ids], dtype=torch.long, device=self.device)
        generated = []
        seen_tokens = set(ids)

        with torch.no_grad():
            # 1. Prefill phase (one-shot parallel forward)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, cache_O_list, state_I_list = self.model(context, return_state=True)

            cache_seq_len = context.shape[1]
            
            # Pre-allocate exact cache for O(1) decoding
            if cache_O_list is not None:
                for i, (k_cache, v_cache) in enumerate(cache_O_list):
                    b, h, seq, d = k_cache.shape
                    new_k = torch.zeros(b, h, seq + max_new_tokens, d, device=k_cache.device, dtype=k_cache.dtype)
                    new_v = torch.zeros(b, h, seq + max_new_tokens, v_cache.shape[-1], device=v_cache.device, dtype=v_cache.dtype)
                    new_k[:, :, :seq, :] = k_cache
                    new_v[:, :, :seq, :] = v_cache
                    cache_O_list[i] = (new_k, new_v)

            # 2. Generation phase
            for step in range(max_new_tokens):
                # --- DEBUG: Check the integrity of the state and cache ---
                if step == 0 and DEBUG_MODE:
                    attn_layer = self.model.layers[0].attn
    
                    # 1. Check out_proj weight NORM (should be ~1.5 - 2.0, not 0.0)
                    norm_val = attn_layer.out_proj.weight.norm().item()
                    print(f"out_proj norm: {norm_val:.4f}")  # Expected: ~1.9
                    
                    # 2. Check exact cache size (should match prompt length + 1)
                    if cache_O_list is not None:
                        cache_len = cache_O_list[0][0].shape[2]
                        print(f"Exact Cache Length: {cache_len} (Prompt length: {context.shape[1]})")
                    
                    # 3. Check GLA State magnitude (should NOT be zero)
                    if state_I_list is not None:
                        state_sum = state_I_list[0].sum().item()
                        state_norm = state_I_list[0].norm().item()
                        print(f"GLA State Sum: {state_sum:.4f}, Norm: {state_norm:.4f}")
                    
                    # 4. [CRITICAL] Check the Mixing Gate manually by computing it on the fly
                    # We need Q and K from the first layer's forward_step output.
                    # Since we can't easily grab Q/K, let's just log the `gamma` (GLA decay).
                    # In forward_step, gamma = torch.sigmoid(-gate_logits). If this is near 1, 
                    # the GLA state decays instantly, meaning it forgets everything.
                    # We can check if the gate_proj bias (-1.0) survived loading.
                    gate_bias = attn_layer.gate_proj.bias
                    print(f"Gate bias mean: {gate_bias.mean().item():.4f}") # Should be ~ -1.0

                    mix_bias = attn_layer.mix_proj.bias
                    print(f"Mix gate bias mean: {mix_bias.mean().item():.4f}")  # If this is < -2.0, mix_g is near 0

                if DEBUG_MODE:
                    # Sanitize logits to prevent CUDA asserts in untrained models (NaNs or Infs)
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print("NaN/Inf detected in logits!")
                        print(f"Logits: {logits}")
                
                logits = logits[:, -1, :] / temperature

                for past_token in seen_tokens:
                    if logits[0, past_token] < 0:
                        logits[0, past_token] *= repetition_penalty
                    else:
                        logits[0, past_token] /= repetition_penalty
                
                if top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                next_token_id = next_token.item()
                if next_token_id == self.tokenizer.eot_token:
                    break
                
                context = torch.cat([context, next_token], dim=1)
                generated.append(next_token_id)
                seen_tokens.add(next_token_id)
                
                if stream:
                    yield next_token_id
                    
                x = next_token
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, cache_O_list, state_I_list = self.model.forward_step(
                        x, cache_O_list, state_I_list, cache_seq_len=cache_seq_len
                    )
                cache_seq_len += 1
                
        if not stream:
            return self.tokenizer.decode(generated)

    @torch.no_grad()
    def get_attention_output(self, input_ids: torch.Tensor):
        """Return the model output (logits) for fidelity comparisons."""
        self.model.eval()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = self.model(input_ids)
        return logits
