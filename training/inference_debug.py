""" meant for debugging """
import os
import torch
import torch.nn.functional as F
import tiktoken
from src.HybridOutlierFactorizedAttentionTrain import SubwordLM
from training.training_config import LanguageModelingExperimentConfig

def do_debug_inference(config: LanguageModelingExperimentConfig):
    device = torch.device(config.device)
    model_name = config.model_name
    checkpoint_path = os.path.join(f"data/training/{model_name}", "checkpoint.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}.")
        return

    print(f"Loading {model_name} raw SubwordLM from checkpoint (debug inference)...")
    
    # Use the same tokenizer as the InferenceEngine
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load the EXACT model class used during training
    model = SubwordLM(config.vocab_size, config.model_config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state = ckpt.get('model_state_dict', ckpt)
    # Remove cached items from state dict if any exist
    state = {k: v for k, v in state.items() if "_cached" not in k}
    
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    
    print("\nModel loaded successfully! Vanilla autoregressive loop (no KV cache) initialized.")
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
            
            input_ids = tokenizer.encode(prompt)
            generated_ids = input_ids.copy()
            
            max_new_tokens = 100
            temperature = 0.8
            top_k = 50
            repetition_penalty = 1.15
            
            prev_text = tokenizer.decode(generated_ids)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    for _ in range(max_new_tokens):
                        # Construct full sequence tensor for vanilla forward pass
                        x = torch.tensor([generated_ids], dtype=torch.long, device=device)
                        
                        logits, _ = model(x)
                        next_token_logits = logits[0, -1, :] / temperature
                        
                        # Repetition penalty
                        for past_token in set(generated_ids):
                            if next_token_logits[past_token] < 0:
                                next_token_logits[past_token] *= repetition_penalty
                            else:
                                next_token_logits[past_token] /= repetition_penalty

                        # Top-k
                        v, _ = torch.topk(next_token_logits, top_k)
                        next_token_logits[next_token_logits < v[-1]] = float('-inf')
                        
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                        
                        if next_token == tokenizer.eot_token:
                            break
                            
                        generated_ids.append(next_token)
                        
                        full_text = tokenizer.decode(generated_ids)
                        new_chunk = full_text[len(prev_text):]
                        print(new_chunk, end="", flush=True)
                        prev_text = full_text
            
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    config = LanguageModelingExperimentConfig()
    do_debug_inference(config)
