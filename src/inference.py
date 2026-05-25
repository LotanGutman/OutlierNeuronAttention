import torch
import torch.nn.functional as F
import tiktoken
import os
from src.triton_model import SubwordLM
from src.config import ModelConfig, TrainingConfig

class InferenceEngine:
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainingConfig, checkpoint_path: str = None):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = torch.device(train_cfg.device)
        self.tokenizer = tiktoken.get_encoding(train_cfg.tokenizer_name)
        self.vocab_size = self.tokenizer.n_vocab

        # Initialize the optimized model
        self.model = SubwordLM(self.vocab_size, model_cfg).to(self.device)
        self.model.eval()

        if checkpoint_path is None:
            # find the latest checkpoint in the checkpoint directory
            checkpoint_path = self._find_latest_checkpoint(train_cfg.checkpoint_dir)
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
            # path[3:] removes "C:\" or "C:/"
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
        
        # Matches files like hybrid_attn_step_500.pt
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
        # Load checkpoint (handling weights_only for security/compatibility)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt['model_state_dict']
        
        # handle missing buffer keys for routing indices (cached outlier/inlier)
        # We ensure the model has placeholders for these so load_state_dict(strict=True) works
        for key, tensor in list(state.items()):
            if '_cached_outlier_idx' in key or '_cached_inlier_idx' in key:
                *path_parts, attr = key.split('.')
                obj = self.model
                for p in path_parts:
                    obj = getattr(obj, p)
                
                # If the buffer is None (initial state), set a placeholder
                if getattr(obj, attr) is None:
                    placeholder = torch.zeros_like(tensor)
                    setattr(obj, attr, placeholder)
                    
        self.model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint from {path} (iter {ckpt.get('iter', 'unknown')})")

    def generate(self, prompt: str = "", max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50):
        """Generate text from a prompt. If prompt is empty, start from eot token."""
        self.model.eval()
        if prompt:
            ids = self.tokenizer.encode(prompt)
        else:
            ids = [self.tokenizer.eot_token]
            
        context = torch.tensor([ids], dtype=torch.long, device=self.device)
        generated = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # crop to block_size (maximum sequence length supported by pos embeddings)
                x = context[:, -self.model_cfg.block_size:]
                
                # Use autocast for bfloat16 inference (matches training precision)
                # Note: Triton kernel internal logic handles float32 accumulation.
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = self.model(x)
                
                # Focus on the last token's logits
                logits = logits[:, -1, :] / temperature
                
                if top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                context = torch.cat([context, next_token], dim=1)
                generated.append(next_token.item())
                
        return self.tokenizer.decode(generated)

    @torch.no_grad()
    def get_attention_output(self, input_ids: torch.Tensor):
        """Return the model output (logits) for fidelity comparisons."""
        self.model.eval()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = self.model(input_ids)
        return logits

# Example usage:
"""
from src.config import ModelConfig, TrainingConfig
from src.inference import InferenceEngine

model_cfg = ModelConfig()
train_cfg = TrainingConfig()
engine = InferenceEngine(model_cfg, train_cfg)

print("Model loaded. Enter a prompt (or 'exit' to quit):")
while True:
    prompt = input(">> ")
    if prompt.lower() in ['exit', 'quit']:
        break
    output = engine.generate(prompt=prompt, max_new_tokens=200, temperature=0.8)
    print(output)
    print("-" * 50)
"""
