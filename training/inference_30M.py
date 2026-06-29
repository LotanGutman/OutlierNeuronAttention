import os
import torch
from src.inference import InferenceEngine
from training.training_configs import LanguageModelingExperimentConfig
from src.config import TrainingConfig

def main():
    config = LanguageModelingExperimentConfig()
    
    # We create a dummy TrainingConfig just for the InferenceEngine to find the directory
    # and know the device
    train_cfg = TrainingConfig(
        device=config.device,
        checkpoint_dir="data/training/30M"
    )
    
    checkpoint_path = os.path.join(train_cfg.checkpoint_dir, "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Please run train_30M.py first.")
        return
        
    print("Loading 30M HOFA model from checkpoint...")
    engine = InferenceEngine(config.model_config, train_cfg, checkpoint_path=checkpoint_path)
    
    print("\nModel loaded successfully! Heterogeneous decoding loop initialized.")
    print("Enter a prompt (or 'exit' to quit):")
    
    while True:
        try:
            prompt = input("\n>> ")
            if prompt.lower() in ['exit', 'quit']:
                break
            if not prompt.strip():
                continue
                
            print(f"\n[Prompt]: {prompt}")
            print("[Generated]:", end=" ", flush=True)
            
            # Stream the generated tokens
            for token in engine.generate(prompt=prompt, max_new_tokens=100, temperature=0.8, top_k=50, stream=True):
                # decode single token. Note: with BPE this may occasionally print
                # replacement chars for partial UTF-8, but it works fine for a smoke test
                chunk = engine.tokenizer.decode([token])
                print(chunk, end="", flush=True)
            
            print() # newline after generation is complete
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError during generation: {e}")

if __name__ == "__main__":
    main()
