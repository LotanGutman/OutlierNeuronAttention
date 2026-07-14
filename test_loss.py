import os
import torch
import math

def inspect_checkpoint(ckpt_path):
    print("=" * 60)
    print(f"Inspecting Metrics for: {ckpt_path}")
    print("=" * 60)
    
    if not os.path.exists(ckpt_path):
        print(f"File not found: {ckpt_path}")
        return
        
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
        
    step = ckpt.get('step', 'Unknown')
    total_tokens = ckpt.get('total_processed_tokens', 'Unknown')
    best_val_ppl = ckpt.get('best_val_ppl', 'Unknown')
    
    print(f"Saved at Step:          {step}")
    if isinstance(total_tokens, (int, float)):
        print(f"Total Processed Tokens: {total_tokens:,}")
    else:
        print(f"Total Processed Tokens: {total_tokens}")
        
    if isinstance(best_val_ppl, (int, float)):
        print(f"Best Val PPL (saved):   {best_val_ppl:.4f} (Loss: {math.log(best_val_ppl):.4f})")
    else:
        print(f"Best Val PPL (saved):   {best_val_ppl}")
        
    metrics = ckpt.get('metrics', {})
    
    if not metrics:
        print("No 'metrics' dictionary found in this checkpoint.")
        print("\n")
        return
        
    # Extract last recorded train loss
    if 'loss' in metrics and len(metrics['loss']) > 0:
        last_train_loss = metrics['loss'][-1]
        print(f"Last Train Loss:        {last_train_loss:.4f}")
    else:
        print("Last Train Loss:        Not recorded")
        
    # Extract last recorded val ppl
    if 'val_ppl' in metrics and len(metrics['val_ppl']) > 0:
        last_val_ppl = metrics['val_ppl'][-1]
        print(f"Last Val PPL:           {last_val_ppl:.4f} (Loss: {math.log(last_val_ppl):.4f})")
    else:
        print("Last Val PPL:           Not recorded")
        
    print("\n")

def main():
    base_dir = "data/training/125M_HOFA"
    
    ckpt_paths = [
        os.path.join(base_dir, "checkpoint.pt"),
        os.path.join(base_dir, "checkpoint_best_val.pt")
    ]
    
    for path in ckpt_paths:
        inspect_checkpoint(path)

if __name__ == "__main__":
    main()
