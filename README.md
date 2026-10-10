<div align="center">

# Hybrid Outlier-Factorized Attention (HOFA)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg?logo=pytorch)](https://pytorch.org)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](docs/preprint.pdf)

**Decoupling associative recall from linear recurrence via channel-wise outlier factorization**

</div>

---

**HOFA** factorizes each attention head's feature dimension into an exact Softmax outlier pathway ($r$ dimensions) for associative recall and a recurrent Gated Linear Attention inlier pathway ($d_h - r$ dimensions) for sub-quadratic sequence processing. A learned token-level mixing gate dynamically blends both streams, preserving full recall capacity while compressing key-value cache footprints.

<p align="center">
  <img src="docs/architecture.png" alt="HOFA Architecture Overview" width="550"/>
</p>

---

## Quickstart

```bash
git clone https://github.com/LotanGutman/OutlierNeuronAttention.git
cd OutlierNeuronAttention
bash install.sh  # or bash install_no_mamba.sh (for environment without mamba-ssm)
```

---

## Reproduction Commands

| Workflow | Command | Description |
| **Data Preparation** | `python main.py train --scale 350M --download-data` (FineWeb-Edu) / `python main.py train --download-data --dataset pg19` (PG19) | Download and tokenize FineWeb-Edu (pretraining) or PG19 (16k continual pretraining) |
| **Pretraining** | `python main.py train --scale 350M` | Run distributed pretraining (`70M`, `125M`, or `350M`) |
| **Continual Pretraining** | `python main.py train --cpt --scale 350M [--model {hofa,mha}]` | Continual pretraining on PG19 at 16k context (HOFA or MHA) |
| **Checkpoints Eval** | `python main.py train --scale 350M --eval` | Evaluate perplexity on validation splits |
| **Context Extension** | `python main.py benchmark --extrapolation` | Training-free extension diagnostics ($1\text{k} \to 16\text{k}$ tokens) |
| **Induction Retrieval** | `python main.py benchmark --induction [--plot]` | Associative recall on synthetic induction heads |
| **Capacity Cliff** | `python main.py benchmark --induction-degradation [--plot]` | Phase transition and critical capacity cliff |
| **Associative Copying** | `python main.py benchmark --copy [--plot]` | Sequence copying capability benchmark |
| **Mass Decomposition** | `python main.py benchmark --keff [--plot]` | $K_{\text{eff}}$ outlier probability mass distribution |
| **Effective Distance** | `python main.py benchmark --distance [--plot]` | Effective attention context distance benchmark |
| **Gate Blending** | `python main.py benchmark --alpha` | Learned mixing gate distribution analysis |
| **Zero-Shot Reasoning** | `python main.py eval --shared --scale 350M` | Zero-shot reasoning benchmarks (ARC, HellaSwag, PIQA) |
| **Kernel Verification** | `python main.py infer --validate` | Verify numerical parity between PyTorch and Triton decode |
| **Decode Profiling** | `python main.py profile --decode --no-use-cache` | Benchmark prefill and decode latency / memory |

---

## Repository Structure

```text
OutlierNeuronAttention/
├── src/                          # Core model layers and Triton kernels
│   ├── HybridOutlierFactorizedAttention.py       # Inference module
│   ├── HybridOutlierFactorizedAttentionTrain.py  # Training model
│   ├── yarn.py                                   # YaRN extrapolation module
│   ├── config.py
│   ├── chunk_gla_inlier.py, exact_attention.py, hofa_decode_triton.py, inference.py  # Fused Triton decode & reference kernels
│   └── modules/                                  # Triton helpers, benchmarking & checkpointing utilities
├── benchmarks/                   # Synthetic, mechanistic, and extrapolation tasks
│   ├── analyze_alpha.py
│   ├── benchmark_K_eff.py
│   ├── benchmark_copying.py
│   ├── benchmark_distance.py
│   ├── benchmark_induction.py
│   ├── benchmark_induction_degradation.py
│   ├── benchmark_long_context.py
│   ├── benchmark_zeroshot.py
│   ├── benchmarks_configs.py
│   ├── profile_decode.py
│   ├── profile_prefill.py
│   ├── validate_kernels.py
│   └── plotting/                                 # Plotting utilities
├── training/                     # Language modeling training pipeline
│   ├── train.py                                  # Distributed pretraining loop
│   ├── download_fineweb.py, download_pg19.py, data_utils.py, training_config.py  # Dataset pipeline & configs
│   └── plot_training.py, inference.py, modules/  # Loss curves & evaluation utilities
├── docs/                         # Technical documentation, preprint PDF, and architecture figures
├── install.sh, install_no_mamba.sh               # Environment setup scripts
├── main.py                       # Unified CLI entrypoint
└── LICENSE
```

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
