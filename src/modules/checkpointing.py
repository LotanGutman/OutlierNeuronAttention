import os
import torch

def save_checkpoint(model, optimizer, metadata: dict, checkpoint_dir: str):
    """
    Saves a single generic checkpoint.
    This overwrites any existing checkpoint in the directory to save space.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metadata': metadata
    }
    
    # Save safely
    temp_path = checkpoint_path + ".tmp"
    torch.save(state, temp_path)
    os.replace(temp_path, checkpoint_path)

def load_checkpoint(model, optimizer, expected_model_name: str, expected_density: int, checkpoint_dir: str):
    """
    Attempts to load a checkpoint from the given directory.
    Returns the metadata dict if successfully loaded and matches the expected model & density.
    Returns None if no checkpoint exists or if the metadata doesn't match.
    """
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    
    if not os.path.exists(checkpoint_path):
        return None
        
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None
        
    metadata = ckpt.get('metadata', {})
    
    if metadata.get('model_name') != expected_model_name:
        return None
        
    if metadata.get('density') != expected_density:
        return None
        
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
    return metadata
