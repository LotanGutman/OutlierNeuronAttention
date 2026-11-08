import os
import gc
import re
import json
import math
import string
import time
import zipfile
from collections import Counter
from typing import Optional, List, Dict, Any, Union
import torch
import tiktoken
from huggingface_hub import hf_hub_download

from src.yarn import YaRNModel, YaRNMode
from training.training_config import YaRNConfig
from benchmarks.benchmarks_configs import LongBenchExperimentConfig, CACHE_PATH

# Official LongBench prompt formats (THUDM/LongBench config/dataset2prompt.json)
LONGBENCH_PROMPTS = {
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
}

# Official LongBench maximum generation lengths (THUDM/LongBench config/dataset2maxlen.json)
LONGBENCH_MAX_NEW_TOKENS = {
    "trec": 64,
    "triviaqa": 32,
    "multifieldqa_en": 64,
}


def _normalize_answer(s: str) -> str:
    """Official LongBench answer normalizer (metrics.py)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _compute_f1(pred: str, target: str) -> float:
    """Official LongBench token-level QA F1 score."""
    p_tokens = _normalize_answer(pred).split()
    t_tokens = _normalize_answer(target).split()
    if not p_tokens or not t_tokens:
        return float(p_tokens == t_tokens)

    common = sum((Counter(p_tokens) & Counter(t_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(p_tokens)
    recall = common / len(t_tokens)
    return (2 * precision * recall) / (precision + recall)


def _classification_score(pred: str, ground_truth: str, all_classes: List[str]) -> float:
    """Official LongBench classification scoring for TREC (metrics.py)."""
    em_match_list = [c for c in all_classes if c in pred]
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def score_sample(task: str, pred: str, answers: List[str], all_classes: Optional[List[str]] = None) -> float:
    """Scores prediction against ground truth answers using official task metric."""
    if not answers:
        return 0.0

    # Official first-line rule
    pred_first_line = pred.lstrip("\n").split("\n")[0].strip()

    if task == "trec":
        classes = all_classes or []
        return max(_classification_score(pred_first_line, gt, classes) for gt in answers)
    return max(_compute_f1(pred_first_line, gt) for gt in answers)


def load_task_samples(task: str, bucket: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads LongBench-E task samples using official word-length bucketing."""
    zip_path = hf_hub_download(repo_id="THUDM/LongBench", filename="data.zip", repo_type="dataset")
    filename = f"data/{task}_e.jsonl"
    raw_lines = []

    with zipfile.ZipFile(zip_path, "r") as z:
        if filename not in z.namelist():
            raise FileNotFoundError(f"Missing {filename} in LongBench data.zip")
        with z.open(filename) as f:
            for line in f:
                raw_lines.append(json.loads(line.decode("utf-8")))

    # Official LongBench-E word count cutoffs: <4000, 4000-8000, >=8000
    filtered = []
    for item in raw_lines:
        w_len = item.get("length", 0)
        if bucket == "0k-4k" and w_len < 4000:
            filtered.append(item)
        elif bucket == "4k-8k" and 4000 <= w_len < 8000:
            filtered.append(item)
        elif bucket == "8k-16k" and w_len >= 8000:
            filtered.append(item)

    return filtered[:limit] if limit and limit > 0 else filtered


def generate_greedy(
    model: YaRNModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int = 50256,
    stop_token_ids: Optional[tuple[int, ...]] = (50256, 198, 628),
) -> List[int]:
    """Greedy decoding that only stops on newline after at least one non-newline token."""
    curr_ids = input_ids.clone()
    generated = []
    started = False

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(max_new_tokens):
            logits, _ = model(curr_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tok_id = next_token.item()
            if tok_id == eos_token_id:
                break
            if stop_token_ids and tok_id in stop_token_ids:
                if started:
                    break
            else:
                started = True
                generated.append(tok_id)
            curr_ids = torch.cat([curr_ids, next_token], dim=-1)

    return generated


def print_longbench_table(results: Dict[str, Any]):
    print("\n" + "=" * 90)
    print(f"{'LongBench-E Results':^90}")
    print("=" * 90)

    header = f"| {'Model':<22} | {'Task':<16} | {'0k-4k':<12} | {'4k-8k':<12} | {'8k-16k':<12} | {'Average':<10} |"
    sep = f"|{'-'*24}|{'-'*18}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*12}|"
    print(header)
    print(sep)

    for m_name, tasks_data in results.items():
        if not isinstance(tasks_data, dict) or m_name == "metadata":
            continue
        for task, b_data in tasks_data.items():
            if not isinstance(b_data, dict):
                continue
            s_0_4 = b_data.get("0k-4k", {}).get("score", float("nan"))
            s_4_8 = b_data.get("4k-8k", {}).get("score", float("nan"))
            s_8_16 = b_data.get("8k-16k", {}).get("score", float("nan"))

            scores = [s for s in (s_0_4, s_4_8, s_8_16) if not math.isnan(s)]
            avg_score = sum(scores) / len(scores) if scores else float("nan")

            str_0_4 = f"{s_0_4 * 100:.2f}%" if not math.isnan(s_0_4) else "-"
            str_4_8 = f"{s_4_8 * 100:.2f}%" if not math.isnan(s_4_8) else "-"
            str_8_16 = f"{s_8_16 * 100:.2f}%" if not math.isnan(s_8_16) else "-"
            str_avg = f"{avg_score * 100:.2f}%" if not math.isnan(avg_score) else "-"

            row = f"| {m_name:<22} | {task:<16} | {str_0_4:<12} | {str_4_8:<12} | {str_8_16:<12} | {str_avg:<10} |"
            print(row)
        print(sep)
    print("=" * 90 + "\n")


def run_longbench_experiment(config: Optional[LongBenchExperimentConfig] = None) -> Dict[str, Any]:
    torch.set_num_threads(4)
    if config is None:
        config = LongBenchExperimentConfig()

    models = [config.models] if isinstance(config.models, YaRNConfig) else list(config.models)
    enc = tiktoken.get_encoding("gpt2")
    device = config.device

    print("=" * 90)
    print("LongBench-E Evaluation (Official Prompts & Metrics)")
    print(f"Device: {device}")
    print(f"Models: {[m.model.model_name for m in models]}")
    print(f"Tasks: {config.tasks}")
    print(f"Buckets: {config.buckets}")
    print(f"Limit per bucket: {config.limit_per_bucket}")
    print(f"Max Context Length: {config.max_seq_len:,}")
    print("=" * 90 + "\n")

    os.makedirs(CACHE_PATH, exist_ok=True)
    cache_file = os.path.join(CACHE_PATH, config.cache_file_name)
    results = {}
    if config.use_cache and not config.force_rerun and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    for yarn_cfg in models:
        m_name = yarn_cfg.model.model_name
        if m_name not in results:
            results[m_name] = {}

        print(f"\n--- Loading Model: {m_name} ---")
        yarn_model = YaRNModel(yarn_cfg, device=device, dtype=torch.bfloat16)
        scale_val = max(16.0, float(config.max_seq_len) / float(yarn_model.original_max_seq_len))
        ymode = getattr(yarn_cfg, "yarn_mode", None) or YaRNMode.default()
        
        effective_mscale = 1.0 + 0.1 * math.log(scale_val)
        yarn_model.set_context_length(config.max_seq_len, scale=scale_val, yarn_mode=ymode)
        yarn_model.eval()

        results.setdefault("metadata", {})[m_name] = {
            "scale": scale_val,
            "effective_mscale": effective_mscale,
            "yarn_mode": str(ymode),
            "max_seq_len": config.max_seq_len,
        }

        for task in config.tasks:
            if task not in results[m_name]:
                results[m_name][task] = {}

            template = LONGBENCH_PROMPTS.get(task, "{context}\n\n{input}")
            max_new = LONGBENCH_MAX_NEW_TOKENS.get(task, config.max_new_tokens)
            stop_ids = (50256, 198, 628)

            for bucket in config.buckets:
                samples = load_task_samples(task, bucket, limit=config.limit_per_bucket)
                if not samples:
                    print(f"[{m_name}] {task} [{bucket}] No samples found.", flush=True)
                    continue

                prev = {} if config.force_rerun else results[m_name][task].get(bucket, {})
                sample_records = list(prev.get("samples", []))
                done_ids = {r["id"] for r in sample_records}
                todo = [(i, s) for i, s in enumerate(samples) if s.get("_id", str(i)) not in done_ids]

                if not todo:
                    print(f"[{m_name}] {task} [{bucket}] Cached: {prev['score'] * 100:.2f}% (N={len(sample_records)})", flush=True)
                    continue

                print(f"\n[{m_name}] {task} [{bucket}] Evaluating {len(todo)}/{len(samples)} samples...", flush=True)
                for idx, sample in todo:
                    context = sample.get("context", "")
                    inp = sample.get("input", "")
                    answers = sample.get("answers", [])
                    all_classes = sample.get("all_classes", None)
                    sample_id = sample.get("_id", str(idx))

                    prompt = template.format(context=context, input=inp)
                    toks = enc.encode(prompt, allowed_special={"<|endoftext|>"})

                    # Truncate context from middle if exceeding max context limit
                    max_prompt_len = config.max_seq_len - max_new
                    is_truncated = len(toks) > max_prompt_len
                    if is_truncated:
                        half = max_prompt_len // 2
                        toks = toks[:half] + toks[-half:]

                    x = torch.tensor([toks], dtype=torch.long, device=device)
                    t_sample = time.time()
                    gen_toks = generate_greedy(yarn_model, x, max_new_tokens=max_new, stop_token_ids=stop_ids)
                    pred_str = enc.decode(gen_toks).strip()

                    sample_score = score_sample(task, pred_str, answers, all_classes=all_classes)
                    dt = time.time() - t_sample

                    sample_records.append({
                        "id": sample_id,
                        "score": sample_score,
                        "pred": pred_str,
                        "answers": answers,
                        "prompt_tokens": len(toks),
                        "truncated": is_truncated,
                    })

                    print(f"  Sample {idx+1:03d}/{len(samples):03d} | Score: {sample_score*100:5.1f}% | Gen: {len(gen_toks):2d} toks | Time: {dt:4.1f}s", flush=True)

                    del x
                    if device == "cuda":
                        torch.cuda.empty_cache()

                    # Save after every sample so an interrupted run resumes here
                    scores = [r["score"] for r in sample_records]
                    results[m_name][task][bucket] = {
                        "score": sum(scores) / len(scores),
                        "num_samples": len(scores),
                        "samples": sample_records,
                    }
                    tmp_file = cache_file + ".tmp"
                    with open(tmp_file, "w") as f:
                        json.dump(results, f, indent=2)
                    os.replace(tmp_file, cache_file)

                print(f"[{m_name}] {task} [{bucket}] => Mean: {results[m_name][task][bucket]['score'] * 100:.2f}% (N={len(sample_records)})\n", flush=True)

        del yarn_model
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    print_longbench_table(results)
    return results
