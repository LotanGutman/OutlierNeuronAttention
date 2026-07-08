import os
import torch
from src.inference import InferenceEngine
from training.training_config import LanguageModelingExperimentConfig

def do_inference(config: LanguageModelingExperimentConfig):
    checkpoint_path = os.path.join(f"data/training/{config.model_name}", "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Please run training for {config.model_name} first.")
        return

    print(f"Loading {config.model_name} HOFA model from checkpoint...")
    engine = InferenceEngine(config, checkpoint_path=checkpoint_path)
    
    print("\nModel loaded successfully! Heterogeneous decoding loop initialized.")
    print("Enter a prompt (or 'exit' to quit):")
    
    while True:
        try:
            prompt = input("\n>> ")
            if prompt.lower() in ['exit', 'quit']:
                break
            if not prompt.strip():
                continue
                
            import time
            start_time = time.time()
            generated_tokens = []
            prev_text = ""
            
            prefill_time = None
            decode_start = None
            
            print("[Compiling...] ", end="", flush=True)
            
            try:
                # Stream the generated tokens
                for token in engine.generate(prompt=prompt, max_new_tokens=100, temperature=0.1, top_k=5, stream=True):
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