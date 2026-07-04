"""
Evaluate a pretrained HOFA/MHA model on HellaSwag using lm-eval (Python API).

Usage:
    from training.training_config import make_125M_hofa
    from benchmarks.benchmarks_configs import EvalExperimentConfig
    from benchmarks.benchmark_swag import evaluate_hellaswag
    
    config = make_125M_hofa()
    eval_config = EvalExperimentConfig(limit=1000)
    evaluate_hellaswag(config, eval_config)
"""

import os
import torch
from dataclasses import asdict
from transformers import (
    AutoConfig, AutoModel, AutoModelForCausalLM,
    PretrainedConfig, PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from src.HybridOutlierFactorizedAttentionTrain import SubwordLM
from src.config import ModelConfig
from training.training_config import LanguageModelingExperimentConfig
import lm_eval
from lm_eval import simple_evaluate
import shutil
from benchmarks.benchmarks_configs import EvalExperimentConfig
# ------------------------------------------------------------------
# 1. Hugging Face wrapper for your custom model
# ------------------------------------------------------------------
class HOFAConfig(PretrainedConfig):
    model_type = "hofa"

    def __init__(self, vocab_size=50257, model_config=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size

        # If model_config is a dict (when loading from saved JSON), convert back to ModelConfig
        if model_config is not None and isinstance(model_config, dict):
            self.model_config = ModelConfig(**model_config)
        else:
            self.model_config = model_config

    def to_dict(self):
        """
        Override to serialise the nested ModelConfig dataclass.
        This is called by save_pretrained() when writing config.json.
        """
        output = super().to_dict()
        if hasattr(self, 'model_config') and self.model_config is not None:
            output['model_config'] = asdict(self.model_config)
        return output


class HOFAModel(PreTrainedModel):
    config_class = HOFAConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = SubwordLM(config.vocab_size, config.model_config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits, loss = self.model(input_ids, targets=labels)
        return CausalLMOutputWithPast(logits=logits, loss=loss)


# Register the custom classes with Hugging Face
AutoConfig.register("hofa", HOFAConfig)
AutoModel.register(HOFAConfig, HOFAModel)
AutoModelForCausalLM.register(HOFAConfig, HOFAModel)


# ------------------------------------------------------------------
# 2. Main evaluation function
# ------------------------------------------------------------------
def evaluate_hellaswag(
    config: LanguageModelingExperimentConfig,
    eval_config: EvalExperimentConfig
):
    """
    Evaluate a pretrained model (specified by `config`) on HellaSwag.

    Args:
        config: LanguageModelingExperimentConfig (e.g., from make_125M_hofa())
        eval_config: EvalExperimentConfig containing limit, device, and seed.
    """
    model_name = config.model_name
    ckpt_path = f"data/training/{model_name}/checkpoint.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 1. Prepare temporary HF model directory
    tmp_dir = f"tmp_hofa_{model_name}"
    os.makedirs(tmp_dir, exist_ok=True)

    # 2. Save config
    hf_config = HOFAConfig(
        vocab_size=config.vocab_size,
        model_config=config.model_config,
    )
    hf_config.save_pretrained(tmp_dir)

    # 3. Load weights into wrapper and save
    model = HOFAModel(hf_config)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.model.load_state_dict(ckpt['model_state_dict'])
    model.save_pretrained(tmp_dir, safe_serialization=False)

    # 4. Run lm_eval with auto batching
    print(f"Evaluating {model_name} on HellaSwag (limit={eval_config.limit}, seed={eval_config.seed})...")
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={tmp_dir},tokenizer=gpt2",
        tasks=["hellaswag"],
        limit=eval_config.limit,
        device=eval_config.device,
        batch_size=eval_config.batch_size,
        numpy_random_seed=eval_config.seed,
        torch_random_seed=eval_config.seed,
        fewshot_random_seed=eval_config.seed,
    )

    # 5. Extract and print metrics (keys have ",none" appended)
    hellaswag = results.get("results", {}).get("hellaswag", {})
    acc = hellaswag.get("acc,none")
    acc_norm = hellaswag.get("acc_norm,none")

    print("\n" + "=" * 60)
    print(f"HellaSwag Results for {model_name}")
    print("=" * 60)
    if acc is not None:
        print(f"Accuracy (exact match): {acc:.4f} ({acc*100:.2f}%)")
    if acc_norm is not None:
        print(f"Accuracy (normalised):  {acc_norm:.4f} ({acc_norm*100:.2f}%)")
    if acc is None and acc_norm is None:
        print("No accuracy metrics found. Raw output:", hellaswag)

    print(f"Samples evaluated: {results.get('config', {}).get('limit', 'full')}")
    print("=" * 60)

    # clean up temporary directory
    shutil.rmtree(tmp_dir)

    return results


# ------------------------------------------------------------------
# 3. Example: run it directly
# ------------------------------------------------------------------
if __name__ == "__main__":
    from training.training_config import make_125M_hofa
    from benchmarks.benchmarks_configs import EvalExperimentConfig

    config = make_125M_hofa()
    eval_config = EvalExperimentConfig(limit=1000) # change to None for full 10k
    evaluate_hellaswag(config, eval_config)