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
                
            print(f"\n[Prompt]: {prompt}")
            print("[Generated]: ", end="", flush=True)
            
            generated_tokens = []
            prev_text = ""
            
            # Stream the generated tokens
            for token in engine.generate(prompt=prompt, max_new_tokens=100, temperature=0.1, top_k=5, stream=True):
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
    config = LanguageModelingExperimentConfig()
    do_inference(config)