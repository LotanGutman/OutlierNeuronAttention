import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os
from src.HybridOutlierFactorizedAttention import SubwordLM
from src.modules.triton_utils import BUCKETS
from src.config import InferenceConfig

DEBUG_MODE = False  # Set to True to enable debug prints

class InferenceEngine:
    def __init__(self, experiment_cfg, inference_cfg=None, checkpoint_path: str = None):
        self.model_cfg = experiment_cfg.model_config
        self.inference_cfg = inference_cfg if inference_cfg is not None else InferenceConfig()
        self.device = torch.device(experiment_cfg.device)
        self.model_name = experiment_cfg.model_name
        self.tokenizer = tiktoken.get_encoding(self.model_cfg.tokenizer_name)
        self.vocab_size = self.tokenizer.n_vocab

        self.model = SubwordLM(self.vocab_size, self.model_cfg).to(self.device)
        self.model.eval()

        if DEBUG_MODE:
            from src.modules.benchmark_utils import patch_attention_for_debugging
            for layer in self.model.layers:
                patch_attention_for_debugging(layer.attn)

        if checkpoint_path is None:
            checkpoint_dir = f"data/training/{self.model_name}"
            checkpoint_path = self._find_latest_checkpoint(checkpoint_dir)
        else:
            checkpoint_path = self._resolve_path(checkpoint_path)
            
        self._load_checkpoint(checkpoint_path)
        self._warmup()

    def _warmup(self):
        if self.inference_cfg is None:
            return
            
        print("\n[INFO] Warming up Triton kernels across all length buckets...")
        
        global DEBUG_MODE
        old_debug = DEBUG_MODE
        DEBUG_MODE = False
        
        for n in BUCKETS:
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = list(self.generate(prompt="x" * min(n, 2000)))
        torch.cuda.synchronize()
        
        DEBUG_MODE = old_debug
        print("[INFO] Warmup complete.")

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


        if DEBUG_MODE or len(missing_keys) > 0 or len(unexpected_keys) > 0:
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
        
        if DEBUG_MODE:
            attn_layer = self.model.layers[0].attn
            norm_val = attn_layer.out_proj.weight.norm().item()
            print(f"--- Static Weights Debug ---")
            print(f"Layer 0 out_proj norm: {norm_val:.4f}")
            gate_bias = attn_layer.gate_proj.bias
            print(f"Layer 0 Gate bias mean: {gate_bias.mean().item():.4f}")
            if hasattr(attn_layer, 'mix_proj'):
                mix_bias = attn_layer.mix_proj.bias
                print(f"Layer 0 Mix gate bias mean: {mix_bias.mean().item():.4f}")
            print(f"----------------------------")
    
    def generate(self, prompt: str = ""):
        """Generate text from a prompt using self.inference_cfg settings."""
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
                for i, cache in enumerate(cache_O_list):
                    if cache is None:
                        continue
                    k_cache, v_cache = cache
                    b, h, seq, d = k_cache.shape
                    new_k = torch.zeros(b, h, seq + self.inference_cfg.max_new_tokens, d, device=k_cache.device, dtype=k_cache.dtype)
                    new_v = torch.zeros(b, h, seq + self.inference_cfg.max_new_tokens, v_cache.shape[-1], device=v_cache.device, dtype=v_cache.dtype)
                    new_k[:, :, :seq, :] = k_cache
                    new_v[:, :, :seq, :] = v_cache
                    cache_O_list[i] = (new_k, new_v)

            # 2. Generation phase
            for step in range(self.inference_cfg.max_new_tokens):
                # --- DEBUG: Check the integrity of the state and cache ---
                if step == 0 and DEBUG_MODE:
                    # 1. Check exact cache size (should match prompt length + 1)
                    if cache_O_list is not None:
                        first_valid_cache = next((c for c in cache_O_list if c is not None), None)
                        if first_valid_cache is not None:
                            cache_len = first_valid_cache[0].shape[2]
                            print(f"Exact Cache Length: {cache_len} (Prompt length: {context.shape[1]})")
                    
                    # 2. Check GLA State and Dynamic Gates
                    state_msg = ""
                    if state_I_list is not None:
                        state_I_tensor = state_I_list[0][0] if isinstance(state_I_list[0], tuple) else state_I_list[0]
                        state_msg = f"GLA Sum: {state_I_tensor.sum().item():.4f}, Norm: {state_I_tensor.norm().item():.4f} | "
                        
                    attn_layer = self.model.layers[0].attn
                    if hasattr(attn_layer, '_last_mix_g'):
                        mix_g_mean = attn_layer._last_mix_g.mean().item()
                        gate_logits_mean = attn_layer._last_gate_logits.mean().item()
                        print(f"{state_msg}Mix Gate (mean): {mix_g_mean:.4f} | GLA Gate Logits: {gate_logits_mean:.4f}")
                    elif state_msg:
                        print(state_msg.rstrip(" | "))

                if DEBUG_MODE:
                    # Sanitize logits to prevent CUDA asserts in untrained models (NaNs or Infs)
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print("NaN/Inf detected in logits!")
                        print(f"Logits: {logits}")
                
                logits = logits[:, -1, :]
                if self.inference_cfg.temperature >= 1e-5:
                    logits = logits / self.inference_cfg.temperature

                # Vectorized repetition penalty (replaces Python for-loop)
                if seen_tokens:
                    seen_idx = torch.tensor(list(seen_tokens), device=self.device, dtype=torch.long)
                    penalty_logits = logits[0, seen_idx]
                    penalty_logits = torch.where(penalty_logits < 0, penalty_logits * self.inference_cfg.repetition_penalty, penalty_logits / self.inference_cfg.repetition_penalty)
                    logits[0, seen_idx] = penalty_logits
                
                if self.inference_cfg.top_k > 0:
                    v, _ = torch.topk(logits, self.inference_cfg.top_k)
                    logits[logits < v[:, [-1]]] = float('-inf')
                        
                probs = F.softmax(logits, dim=-1)
                if self.inference_cfg.temperature < 1e-5:
                    next_token_id = torch.argmax(logits).item()
                else:
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
                            
                if next_token_id == self.tokenizer.eot_token:
                    break
                
                generated.append(next_token_id)
                seen_tokens.add(next_token_id)
                
                if self.inference_cfg.stream:
                    yield next_token_id
                    
                x = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, cache_O_list, state_I_list = self.model.forward_step(
                        x, cache_O_list, state_I_list, cache_seq_len=cache_seq_len
                    )
                cache_seq_len += 1
                
        if not self.inference_cfg.stream:
            return self.tokenizer.decode(generated)

    @torch.no_grad()
    def get_attention_output(self, input_ids: torch.Tensor):
        """Return the model output (logits) for fidelity comparisons."""
        self.model.eval()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = self.model(input_ids)
        return logits

# Simpler engine that uses the Train model (for debugging, to show that the new inference engine works)
# It's heavily simplified (no KV cache, no GLA states, etc.) for pure simplicity (and not speed).
class DebugInferenceEngine:
    def __init__(self, experiment_cfg, inference_cfg=None, checkpoint_path: str = None):
        self.model_cfg = experiment_cfg.model_config
        self.inference_cfg = inference_cfg if inference_cfg is not None else InferenceConfig()
        self.device = torch.device(experiment_cfg.device)
        self.model_name = experiment_cfg.model_name
        self.tokenizer = tiktoken.get_encoding(self.model_cfg.tokenizer_name)
        self.vocab_size = self.tokenizer.n_vocab

        self.model = SubwordLM(self.vocab_size, self.model_cfg).to(self.device)
        self.model.eval()

        if checkpoint_path is None:
            checkpoint_dir = f"data/training/{self.model_name}"
            checkpoint_path = self._find_latest_checkpoint(checkpoint_dir)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        print(f"Loaded checkpoint from {checkpoint_path}")
        
    def _find_latest_checkpoint(self, checkpoint_dir: str) -> str:
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        return os.path.join(checkpoint_dir, "checkpoint.pt")

    def generate(self, prompt: str = ""):
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt)
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                for _ in range(self.inference_cfg.max_new_tokens):
                    x = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
                    
                    logits, _ = self.model(x)
                    next_token_logits = logits[0, -1, :]
                    if self.inference_cfg.temperature >= 1e-5:
                        next_token_logits = next_token_logits / self.inference_cfg.temperature
                    
                    # Repetition penalty
                    for past_token in set(generated_ids):
                        if next_token_logits[past_token] < 0:
                            next_token_logits[past_token] *= self.inference_cfg.repetition_penalty
                        else:
                            next_token_logits[past_token] /= self.inference_cfg.repetition_penalty

                    # Top-k
                    if self.inference_cfg.top_k > 0:
                        v, _ = torch.topk(next_token_logits, self.inference_cfg.top_k)
                        next_token_logits[next_token_logits < v[-1]] = float('-inf')
                    
                    if self.inference_cfg.temperature < 1e-5:
                        next_token = torch.argmax(next_token_logits).item()
                    else:
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                    
                    if next_token == self.tokenizer.eot_token:
                        break
                        
                    generated_ids.append(next_token)
                    
                    if self.inference_cfg.stream:
                        yield next_token
                        
        if not self.inference_cfg.stream:
            return self.tokenizer.decode(generated_ids[len(input_ids):])
