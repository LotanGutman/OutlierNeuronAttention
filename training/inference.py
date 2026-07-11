import os
import torch
import time
from src.inference import InferenceEngine, DebugInferenceEngine
from src.config import InferenceConfig
from training.training_config import LanguageModelingExperimentConfig

def do_inference(config: LanguageModelingExperimentConfig, inference_cfg: InferenceConfig, use_debug: bool = False):
    checkpoint_path = os.path.join(f"data/training/{config.model_name}", "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Please run training for {config.model_name} first.")
        return

    if use_debug:
        print(f"Loading {config.model_name} raw SubwordLM from checkpoint (debug inference)...")
        engine = DebugInferenceEngine(config, inference_cfg=inference_cfg, checkpoint_path=checkpoint_path)
        print("\nModel loaded successfully! Vanilla autoregressive loop (no KV cache) initialized.")
    else:
        print(f"Loading {config.model_name} HOFA model from checkpoint...")
        engine = InferenceEngine(config, inference_cfg=inference_cfg, checkpoint_path=checkpoint_path)
        print("\nModel loaded successfully! Heterogeneous decoding loop initialized.")
    if inference_cfg.seed != -1:
        torch.manual_seed(inference_cfg.seed)
        seed_str = f" | Seed: {inference_cfg.seed}"
    else:
        seed_str = ""
        
    print(f"Params: Temp: {inference_cfg.temperature} | Top-k: {inference_cfg.top_k}{seed_str}")
    print("Type 'help' to see available commands.")
    print("Enter a prompt (or 'exit' to quit):")
    
    while True:
        try:
            prompt = input("\n>> ")
            if prompt.lower() in ['exit', 'quit']:
                break
            if not prompt.strip():
                continue
            
            if prompt.lower() in ['help', '/help']:
                print("\n[Available Commands]")
                print("  /temp <float>   : Set the generation temperature (e.g., /temp 0.8)")
                print("  /top_k <int>    : Set the generation top_k (e.g., /top_k 50)")
                print("  /seed <int>     : Set the generation seed (e.g., /seed 42). Use -1 for random.")
                print("  exit, quit      : Exit the inference loop")
                continue
                
            if prompt.startswith("/temp "):
                try:
                    new_temp = float(prompt.split()[1])
                    inference_cfg.temperature = new_temp
                    print(f"[Config] Temperature set to {new_temp}")
                except ValueError:
                    print("[Error] Invalid temperature format. Use: /temp 0.8")
                continue
                
            if prompt.startswith("/top_k "):
                try:
                    new_topk = int(prompt.split()[1])
                    inference_cfg.top_k = new_topk
                    print(f"[Config] Top_k set to {new_topk}")
                except ValueError:
                    print("[Error] Invalid top_k format. Use: /top_k 50")
                continue
                
            if prompt.lower().startswith("/seed "):
                try:
                    new_seed = int(prompt.split()[1])
                    inference_cfg.seed = new_seed
                    if new_seed == -1:
                        torch.seed()
                        print("[Config] Seed set to -1 (pure random)")
                    else:
                        torch.manual_seed(new_seed)
                        print(f"[Config] Seed set to {new_seed}")
                except ValueError:
                    print("[Error] Invalid seed format. Use: /seed 42 or /seed -1")
                continue
                
            start_time = time.time()
            generated_tokens = []
            prev_text = ""
            
            prefill_time = None
            decode_start = None
            
            print("[Prefilling...] ", end="", flush=True)
            
            try:
                # Stream the generated tokens
                for token in engine.generate(prompt=prompt):
                    if prefill_time is None:
                        decode_start = time.time()
                        prefill_time = decode_start - start_time
                        print("\r\033[K[Generated]: ", end="", flush=True)
                        
                    generated_tokens.append(token)
                    full_text = engine.tokenizer.decode(generated_tokens)
                    new_chunk = full_text[len(prev_text):]
                    print(new_chunk, end="", flush=True)
                    prev_text = full_text
            except KeyboardInterrupt:
                print("\n[Generation Interrupted]")
            
            end_time = time.time()
            decode_time = end_time - decode_start if decode_start else 0.0
            num_tokens = len(generated_tokens)
            
            prompt_tokens = len(engine.tokenizer.encode(prompt)) if prompt else 1
            prefill_tps = prompt_tokens / prefill_time if prefill_time and prefill_time > 0 else 0.0
            decode_tps = num_tokens / decode_time if decode_time and decode_time > 0 else 0.0
            
            print() # newline after generation is complete
            print(f"[Speed] Prefill: {prefill_tps:.2f} t/s ({prompt_tokens} tokens in {prefill_time:.2f}s) | Decode: {decode_tps:.2f} t/s ({num_tokens} tokens in {decode_time:.2f}s)")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError during generation: {e}")

if __name__ == "__main__":
    config = LanguageModelingExperimentConfig()
    do_inference(config)