import torch
import os
import numpy as np

from src.modules.benchmark_utils import GenericBenchmarkLM
from benchmarks.benchmarks_configs import InductionVocabExperimentConfig
from benchmarks.benchmark_induction import train_induction

def run_induction_vocab_experiment():
    config = InductionVocabExperimentConfig()
    device = config.device
    torch.manual_seed(config.seed)
    
    models_to_test = config.models_to_test
    vocab_sizes = config.vocab_sizes
    
    seq_len = config.seq_len
    print(f"\n========================================")
    print(f" Starting Vocabulary Size Sweep (Seq Len: {seq_len})")
    print(f"========================================")
    
    for vocab_size in vocab_sizes:
        config.vocab_size = vocab_size
        print(f"\n{'='*40}")
        print(f" EVALUATING VOCAB SIZE: {vocab_size}")
        print(f"{'='*40}")
        
        for name, attn_type, r_val in models_to_test:
            # Resetting the seed here ensures every model sees the *exact same* 
            # sequence of training data, providing the fairest possible comparison.
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            
            if r_val is not None:
                config.model_config.r = r_val
                
            model = GenericBenchmarkLM(
                vocab_size=config.vocab_size,
                d_model=config.model_config.d_model,
                attn_type=attn_type,
                num_heads=config.model_config.num_heads,
                num_layers=config.model_config.num_layers,
                model_cfg=config.model_config
            ).to(device)

            # Isolated checkpoint directory
            checkpoint_dir = f"data/induction_vocab_models/seqlen_{seq_len}/vocab_{vocab_size}/induction_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            history = train_induction(model, config, name, checkpoint_dir=checkpoint_dir)
            
            # Save final model state dict for easy loading later
            torch.save(model.state_dict(), f"{checkpoint_dir}/final_model.pt")
            
            del model
            torch.cuda.empty_cache()
