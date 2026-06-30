import os
from matplotlib.pyplot import text
import torch
from src.inference import InferenceEngine
from training.training_config import LanguageModelingExperimentConfig
from src.config import TrainingConfig

def do_inference(model_name: str = "30M"):
    config = LanguageModelingExperimentConfig()
    
    # We create a dummy TrainingConfig just for the InferenceEngine to find the directory
    # and know the device
    train_cfg = TrainingConfig(
        device=config.device,
        checkpoint_dir=f"data/training/{model_name}"
    )
    
    checkpoint_path = os.path.join(train_cfg.checkpoint_dir, "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Please run train.py for {model_name} first.")
        return
        
    print(f"Loading {model_name} HOFA model from checkpoint...")
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
            print("[Generated]: ", end="", flush=True)
            
            generated_tokens = []
            prev_text = ""
            
            # Stream the generated tokens
            for token in engine.generate(prompt=prompt, max_new_tokens=100, temperature=0.8, top_k=50, stream=True):
                generated_tokens.append(token)

                full_text = engine.tokenizer.decode(generated_tokens)

                new_chunk = full_text[len(prev_text):]
                print(new_chunk, end="", flush=True)

                prev_text = full_text
            
            print() # newline after generation is complete
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError during generation: {e}")

if __name__ == "__main__":
    do_inference()