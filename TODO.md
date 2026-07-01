# HOFA Project TODOs

## Benchmarking & Evaluation
- [x] **Induction Head `r=12`:** Add `r = 12` to the induction head benchmark and verify that it fails. (The bound predicts 13 bits are required for $V=8192$, so $r=12$ should theoretically collapse).
- [x] **Induction Head Scaling:** Fix the induction head task scaling issue. It currently crashes at 128 context length, which suggests a potential architectural or memory management bug in the implementation.

- [x] **Language Model Distance Metric:** Implemented effective attention distance (Child et al. 2019) as `benchmarks/compute_distance_metric.py` and long-distance perplexity as `benchmarks/compute_long_distance_ppl.py`. Both use the 30M HOFA checkpoint.
- [ ] **Trained MHA Baseline for Distance Metric:** The long-distance perplexity and effective distance metrics currently only compare against an untrained MHA. Need a trained MHA checkpoint at comparable scale (30M/70M) for a meaningful comparison — the whole point is measuring whether HOFA preserves real trained attention patterns vs. MHA. Add MHA training configs and re-run both distance scripts on the MHA checkpoint.
- [ ] 4. Run `profile.py` for throughput/latency scaling across sequence lengths up to 131k.
- [ ] **Ablation Studies:** Re-run baseline comparisons against standard Gated Linear Attention models to quantify the exact perplexity and recall improvements gained by adding the exact-routing $r$-subspace.

## Architecture & Kernel Optimization
- [ ] **Custom Triton Backward Kernel:** Investigate numerical instability (division by near-zero $\gamma$) in the custom GLA backward kernel under `bfloat16`. Fix it so we don't have to rely exclusively on `fla.ops.gla.chunk` for training.
- [ ] **Heterogeneous Routing:** Experiment with dynamic or learned routing allocations instead of statically hardcoding the $r = [64, 32, 16, \dots]$ dimensions across layers.
- [ ] **Scaling Laws:** Test the architecture stability and perplexity improvements at larger scales (>125M parameters) and extreme context lengths (1M+ tokens) to empirically validate the linear memory scaling claims.
