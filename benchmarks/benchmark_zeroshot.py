"""
Evaluate pretrained models on Zero-Shot Common-Sense Reasoning benchmarks.

Usage:
    from training.training_config import make_125M_HOFA
    from benchmarks.benchmarks_configs import EvalExperimentConfig
    from benchmarks.benchmark_zeroshot import evaluate_zeroshot
    
    config = make_125M_HOFA()
    eval_config = EvalExperimentConfig(limit=1000)
    evaluate_zeroshot(config, eval_config)
"""

import os
import torch
from dataclasses import asdict
from typing import Union, List
from transformers import (
    AutoConfig, AutoModel, AutoModelForCausalLM,
    PretrainedConfig, PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from src.HybridOutlierFactorizedAttentionTrain import SubwordLM
from src.config import ModelConfig
from training.training_config import LanguageModelingExperimentConfig
from benchmarks.benchmarks_configs import EvalExperimentConfig, CACHE_PATH
from lm_eval import simple_evaluate
import shutil

# ------------------------------------------------------------------
# 1. Hugging Face wrapper for your custom model
# ------------------------------------------------------------------
class HOFAConfig(PretrainedConfig):
    model_type = "hofa"

    def __init__(self, vocab_size=50257, model_config=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size

        if model_config is not None and isinstance(model_config, dict):
            self.model_config = ModelConfig(**model_config)
        else:
            self.model_config = model_config

    def to_dict(self):
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


# Register the custom classes
AutoConfig.register("hofa", HOFAConfig)
AutoModel.register(HOFAConfig, HOFAModel)
AutoModelForCausalLM.register(HOFAConfig, HOFAModel)

# ------------------------------------------------------------------
# 2. Main evaluation function
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Helper: Print formatted comparison table
# ------------------------------------------------------------------
def print_zeroshot_table(all_results: dict, print_tasks: tuple, title: str = "Zero-Shot Reasoning Benchmark Results"):
    if not all_results:
        print("No results to display.")
        return

    model_col_width = max(25, max(len(str(m)) for m in all_results.keys()))
    total_width = model_col_width + 5 + 15 * len(print_tasks)

    print("\n" + "=" * total_width)
    print(title)
    print("=" * total_width)

    # Header
    header = f"| {'Model':<{model_col_width}} |"
    for task in print_tasks:
        header += f" {task:<12} |"
    print(header)
    
    # Separator
    separator = f"|{'-'*(model_col_width+2)}|"
    for task in print_tasks:
        separator += f"{'-'*14}|"
    print(separator)

    # Data Rows
    for model_name, metrics in all_results.items():
        if not isinstance(metrics, dict):
            continue
        row = f"| {model_name:<{model_col_width}} |"
        for task in print_tasks:
            task_metrics = metrics.get(task, {})
            val = task_metrics.get("acc_norm,none")
            if val is None:
                val = task_metrics.get("acc,none")
            
            if val is not None:
                val_str = f"{val*100:.2f}%"
            else:
                val_str = "-"
            
            row += f" {val_str:<12} |"
        print(row)
    
    print("=" * total_width + "\n")


# ------------------------------------------------------------------
# 2. Main evaluation function for trained local checkpoints
# ------------------------------------------------------------------
def evaluate_zeroshot(
    configs: Union[LanguageModelingExperimentConfig, List[LanguageModelingExperimentConfig]],
    eval_config: EvalExperimentConfig,
    is_simple: bool = False
):
    assert eval_config.batch_size == 1, "batch_size must be 1 to prevent padding corruption in unmasked SubwordLM models."

    if not isinstance(configs, list):
        configs = [configs]

    # If simple mode, override tasks to only run HellaSwag
    requested_tasks = ("hellaswag",) if is_simple else eval_config.tasks
        
    os.makedirs(CACHE_PATH, exist_ok=True)
    all_results = {}

    for config in configs:
        model_name = config.model_name
        print(f"\nProcessing zero-shot evaluation for {model_name}...")
        
        # 1. Identify which checkpoint to use
        ckpt_path = f"data/training/{model_name}/checkpoint_best_val.pt"
        ckpt_type = "best validation"
        if not os.path.exists(ckpt_path):
            ckpt_path = f"data/training/{model_name}/checkpoint.pt"
            ckpt_type = "latest training"
            
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for {model_name} at {ckpt_path}. Skipping.")
            continue

        print(f"Using {ckpt_type} checkpoint for {model_name}: {ckpt_path}")
        current_mtime = os.path.getmtime(ckpt_path)

        # 2. Check the cache
        cache_filename = f"{model_name}_zeroshot_cache.pt"
        cache_path = os.path.join(CACHE_PATH, cache_filename)

        model_metrics = {}
        if eval_config.use_cache and not eval_config.force_rerun and os.path.exists(cache_path):
            try:
                cached_data = torch.load(cache_path, weights_only=False)
                if cached_data.get("_mtime") == current_mtime:
                    model_metrics = cached_data
                    tasks_in_cache = [k for k in model_metrics.keys() if k != "_mtime"]
                    print(f"Loaded valid cache from {cache_path} with tasks: {tasks_in_cache}")
                else:
                    print(f"Cache for {model_name} is stale (checkpoint was modified). Starting fresh.")
            except Exception as e:
                print(f"Failed to load cache: {e}. Starting fresh.")

        # Determine which tasks actually need to be run
        tasks_to_run = [t for t in requested_tasks if t not in model_metrics]
        
        if len(tasks_to_run) == 0:
            print(f"All requested tasks are already cached for {model_name}. Skipping lm-eval.")
            all_results[model_name] = model_metrics
            continue

        # 3. We need to evaluate the missing tasks
        # Prepare temporary HF model directory
        tmp_dir = f"tmp_hofa_{model_name}"
        os.makedirs(tmp_dir, exist_ok=True)

        # Save config
        hf_config = HOFAConfig(
            vocab_size=config.vocab_size,
            model_config=config.model_config,
        )
        hf_config.save_pretrained(tmp_dir)

        # Load weights into wrapper and save
        model = HOFAModel(hf_config)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.model.load_state_dict(ckpt['model_state_dict'])
        model.save_pretrained(tmp_dir, safe_serialization=False)

        # 4. Run lm_eval on MISSING tasks only
        print(f"Running lm-eval on {model_name} for MISSING tasks: {tasks_to_run}")
        print(f"(limit={eval_config.limit}, seed={eval_config.seed}, batch_size={eval_config.batch_size})")
        
        results = simple_evaluate(
            model="hf",
            model_args=f"pretrained={tmp_dir},tokenizer=gpt2",
            tasks=tasks_to_run,
            limit=eval_config.limit,
            device=eval_config.device,
            batch_size=eval_config.batch_size,
            numpy_random_seed=eval_config.seed,
            torch_random_seed=eval_config.seed,
            fewshot_random_seed=eval_config.seed,
        )

        # Merge new results with existing cached results
        new_metrics = results.get("results", {})
        for t, m in new_metrics.items():
            model_metrics[t] = m
            
        # Store checkpoint mtime to detect stale cache
        model_metrics["_mtime"] = current_mtime
        
        all_results[model_name] = model_metrics

        # Save combined metrics back to PyTorch cache
        torch.save(model_metrics, cache_path)
        print(f"Saved merged results to cache: {cache_path}")
        
        # Clean up the temporary huggingface directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print_zeroshot_table(all_results, requested_tasks, title="Zero-Shot Reasoning Benchmark Results")


# ------------------------------------------------------------------
# 3. Pretrained internet models zero-shot evaluation function
# ------------------------------------------------------------------
PRETRAINED_BASELINE_MODELS = {
    "125M": [
        "gpt2",
        "EleutherAI/pythia-160m",
        "HuggingFaceTB/SmolLM-135M",
        "state-spaces/mamba-130m",
    ],
    "350M": [
        "facebook/opt-350m",
        "EleutherAI/pythia-410m",
        "HuggingFaceTB/SmolLM-360M",
        "state-spaces/mamba-370m",
    ],
}

def evaluate_pretrained_zeroshot(
    eval_config: EvalExperimentConfig,
    is_simple: bool = False,
    models: List[str] = None,
    scale: str = "125M"
):
    if models is None:
        models = PRETRAINED_BASELINE_MODELS.get(scale, PRETRAINED_BASELINE_MODELS["125M"])

    requested_tasks = ("hellaswag",) if is_simple else eval_config.tasks
    os.makedirs(CACHE_PATH, exist_ok=True)
    cache_path = os.path.join(CACHE_PATH, "pretrained_models_zeroshot_cache.pt")

    all_cached_results = {}
    if eval_config.use_cache and not eval_config.force_rerun and os.path.exists(cache_path):
        try:
            all_cached_results = torch.load(cache_path, weights_only=False)
            print(f"Loaded pretrained models cache from {cache_path}")
        except Exception as e:
            print(f"Failed to load pretrained cache: {e}. Starting fresh.")

    for model_name in models:
        print(f"\nProcessing zero-shot evaluation for pretrained model '{model_name}'...")
        model_metrics = all_cached_results.get(model_name, {})

        tasks_to_run = [t for t in requested_tasks if t not in model_metrics]

        if len(tasks_to_run) == 0:
            print(f"All requested tasks are already cached for {model_name}. Skipping lm-eval.")
            continue

        print(f"Running lm-eval on pretrained '{model_name}' for MISSING tasks: {tasks_to_run}")
        print(f"(limit={eval_config.limit}, seed={eval_config.seed}, batch_size={eval_config.batch_size})")

        # Determine if model requires native mamba_ssm backend
        is_mamba = "mamba" in model_name.lower() and not model_name.endswith("-hf")
        eval_model_type = "mamba_ssm" if is_mamba else "hf"

        try:
            results = simple_evaluate(
                model=eval_model_type,
                model_args=f"pretrained={model_name}",
                tasks=tasks_to_run,
                limit=eval_config.limit,
                device=eval_config.device,
                batch_size=eval_config.batch_size,
                numpy_random_seed=eval_config.seed,
                torch_random_seed=eval_config.seed,
                fewshot_random_seed=eval_config.seed,
            )
            new_metrics = results.get("results", {})
            for t, m in new_metrics.items():
                model_metrics[t] = m

            all_cached_results[model_name] = model_metrics
            torch.save(all_cached_results, cache_path)
            print(f"Saved merged results for {model_name} to cache: {cache_path}")
        except Exception as e:
            print(f"❌ Error evaluating pretrained model {model_name}: {e}")

    print_zeroshot_table(all_cached_results, requested_tasks, title="Pretrained Baseline Models Zero-Shot Results")