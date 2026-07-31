# HOFA — Path to Submission

Solo, single rented GPU (4090-class). Plan runs sequentially where compute-bound,
in parallel where possible (eval-only work can overlap with training runs).

---

## Phase 0 — Setup (Day 0–1)

- [ ] Freeze `main.pdf` as ground truth. Do not touch its numbers.
- [ ] Set up eval harness once, reused everywhere below: FineWeb-Edu validation split,
      1,000 sequences, batch as before, discard first 5 tokens (attention-sink artifact,
      per Sec 5.11 precedent).
- [ ] Confirm 4090 rental throughput (tokens/sec) on your existing 125M config.
      Use this to estimate wall-clock for every run below before committing.

---

## Phase 1 — Eval-Only Checkpoints & Scaling Heatmaps (Day 1–3)

Eval-only, no training. Do this first — it's cheap and can run while you set up
Phase 2's training configs.

### ~~Task A: Effective Attention Distance on existing 125M checkpoints~~
- [x] Hook post-softmax `p_ij` from HOFA's outlier pathway (Eq. 8), per layer, per head.
- [x] Derive GLA-effective distance approximation from gate decay `γ_t` (inlier pathway).
- [x] Compute same for MHA baseline (trivial — direct softmax).
- [x] Compute same for pure GLA baseline (comparison reference).
- [x] Average `d_i` per layer → one curve per model (+ optional per-head spread bands).
- [x] Plot: layer index (x) vs. avg distance (y), lines for MHA / HOFA-outlier /
      HOFA-GLA-effective / pure GLA.
- [x] Write up findings honestly — including if outlier curve is shorter/noisier
      than MHA's. This is a genuine diagnostic, not a foregone conclusion.

### ~~Task B: The 2x4 Attention Mass Scaling Grid~~
- [x] Download `EleutherAI/pythia-1.4b` and `EleutherAI/pythia-2.8b` to complete the Pure MHA scaling row.
- [x] Download `meta-llama/Llama-3.1-8B` and `Qwen/Qwen2.5-7B` (or `mistralai/Mistral-7B-v0.3`) to complete the GQA scaling row.
- [x] Run the existing Probability Mass Decomposition script ($C(r)$) on these 4 new models over 100 validation sequences.
- [x] Generate the upgraded 2x4 `heatmap_cumsum_normalized.pdf` showing the Constant-$r$ Scaling Law holding across both MHA and GQA architectures up to 8B parameters.

**Deliverables:**
1. New Effective Attention Distance figure + short subsection for the paper.
2. ~~Upgraded 2x4 Attention Mass Heatmap figure (`heatmap_cumsum_normalized.pdf`) to replace existing heatmap.~~

---

## Phase 2 — Ablation sprint at 70M scale (Day 3–10)

8 layers, d_model=512, H=8, 2.8B tokens, same optimizer/schedule as existing config.
**Dropped:** front-loaded inter-layer baseline (HOFA-style first half / pure GLA
second half) — not worth the run.

| # | Run | Purpose |
|---|---|---|
| 1 | 70M MHA baseline | Control |
| 2 | 70M HOFA, heterogeneous $r$ | The proposed architecture |
| 3 | 70M HOFA, flat $r=16$ | Proves the heterogeneous schedule is necessary |
| 4 | 70M Depth-axis (Jamba-style) | Proves channel-axis beats sequential |
| 5 | 70M Width-axis (Hymba-style) | Proves channel-axis beats parallel |
| 6 | 70M HOFA, fixed blend weight | Proves the dynamic $\alpha_h$ gate earns its parameters |

- [x] Run all 6 configs to completion (2.8B tokens each).
- [x] Metrics for all 6: final val perplexity.
- [ ] Metrics for all 6: zero-shot suite (ARC-e/c, PIQA, Winogrande, OBQA, HellaSwag)
      — fills the blank 70M rows in the existing zero-shot table.
- [ ] Compare #2 vs #3 → **pick winning r-schedule for the 350M run.**
- [ ] Compare #2 vs #4 vs #5 → this is your primary defense against "why not just
      alternate/split heads instead of channel-decompose" — write this up as its
      own ablation section.
- [ ] Compare #2 vs #6 → validates the dynamic mixing gate.
- [ ] Check: if ablation results are not 100% conclusive, add a flat `r=16` run.

**Deliverable:** ablation table (6 rows × val PPL + 6 benchmarks), new "Architectural
Ablations" section, r-schedule decision locked for Phase 3.

---

## Phase 3 — Decision gate (Day 10–11)

- [ ] Finalize r-schedule for 350M based on Phase 2 results (heterogeneous vs flat).
- [ ] Recompute wall-clock/cost estimate for 350M @ 14B tokens using confirmed
      throughput numbers.
- [ ] Decide checkpoint cadence (every 1–2B tokens) so a bad run can be caught early
      without losing everything.

---

## Phase 4 — 350M training (Day 11–?, compute-bound)

- [ ] Train **350M HOFA** — 24 layers, d_model=1024, H=8, 14B tokens (40× Chinchilla
      ratio, matching existing config), chosen r-schedule from Phase 3.
- [ ] Train **350M MHA baseline** — same tokens, same budget.
- [ ] Monitor loss curves live; kill/restart early if divergence or NaNs appear
      rather than discovering it after days of spend.
- [ ] Checkpoint regularly per Phase 3 cadence.

**No ablations repeated at this scale** — Phase 2 already validates the architectural
choice; 350M is scale-confirmation only, consistent with the paper's own scoping.

---

## Phase 5 — 350M evaluation (Day ?+1 to ?+3, fast — reuses existing scripts)

- [ ] Val perplexity → fills blank 350M row in perplexity table.
- [ ] Zero-shot suite (ARC-e/c, PIQA, Winogrande, OBQA, HellaSwag) → fills blank
      350M row in zero-shot table.
- [ ] Mixing gate α_h violin plot for 350M (repeat Fig 10/13 analysis) — check if
      soft-blending pattern and Layer-1 anomaly persist at scale.
- [ ] Effective attention distance (Phase 1 method) rerun on 350M checkpoints —
      check if the per-layer curve shape holds at scale.
- [ ] Rerun hardware profiling (KV-cache, decoding/prefill speedup) at 350M's
      dimensions if not already covered by existing Fig 1/3/4 (only if the head
      config differs meaningfully from the profiled 125M-matched dims).

**Deliverable:** all remaining TBD table cells filled with real 350M numbers.

---

## Phase 6 — Write-up & submission prep (final days)

- [ ] Merge new ablation section into paper.
- [ ] Merge effective-attention-distance figure/subsection (125M + 350M).
- [ ] Fill remaining TBDs across all tables.
- [ ] Sanity-pass: every number in abstract/intro matches the canonical benchmark run.
- [ ] Keep GQA/MLA/YaRN/CLA compatibility sections as-is — clearly scoped as
      architectural derivation, not empirically validated. No new experiments needed.
- [ ] Final read-through for internal consistency (no more than one number per claim
      across the whole document).

---

## Explicitly out of scope (do not do)

- Distance-Segmented Perplexity — redundant with induction-head / block-copying
  results, dropped in favor of effective attention distance.
- Front-loaded inter-layer baseline (half HOFA / half GLA) — dropped, low value.
- r-sweep at 350M scale — already covered at small scale + cross-model attention-mass
  heatmaps.
- HOFA-GQA / HOFA-MLA joint training — stays as future work, not this paper.
- Multi-seed runs — not feasible solo on rented compute; note single-seed honestly.
