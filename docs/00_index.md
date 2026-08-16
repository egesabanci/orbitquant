# OrbitQuant Autoresearch — Findings Index

Session: **Make the original TurboQuant better and lighter** (statistical regime,
pure NumPy, no model/GPU). Every experiment gets **one single-responsibility
markdown doc**. Each is reviewable in isolation.

## Metrics (primary first; ↓ = lower is better)

| Metric | Meaning |
|---|---|
| `tq_mse` (primary) | mean per-vector reconstruction MSE over d∈{64,128} × b∈{1,2} grid, realistic norms, honest 8-bit log norm header (↓) |
| `tq_bytes_b2` | honest true bits/token at b=2/d=64 via P0.5 accounting (↓, lightness) |
| `tq_bias_b1` | \|inner-product bias ratio − 2/π\| at b=1 (↓, estimator fidelity) |
| `tq_var_b2` | product-estimator variance / Thm-2 bound (↓, estimator fidelity) |
| `fwht_us` | best-of-N apply time of 3-round FWHT at d=128 in µs (↓, compute) |

## Document list

<!-- generated in order -->.

- **[01 — Baseline measurement](01_baseline.md)** — the starting point every
  experiment is compared against.
- **[02 — Vectorized FWHT butterfly (compute lightness)](02_fwht_vectorization.md)**
  — fwht_us 389 → 46 µs (8.4×), bitwise-identical output, all quality metrics
  unchanged. KEPT.
- **[03 — 6-bit log norm header (side-info lightness)](03_norm_header_6bit.md)**
  — tq_bytes_b2 136 → 134 (1.5%), tq_mse +0.04%. Sweep shows 8 bits waste
  ~2 bits/token; 6-bit sweet spot. KEPT.
- **[04 — Empirical codebooks for hdhdh — negative](04_empirical_codebook_hdhdh_negative.md)**
  — empirical LLB converges to exact-Beta (±0.0001), ~0% gain; exact-Beta
  already optimal for the structured rotation. No code change.
- **[05 — hdhdh is a drop-in for Haar on realistic vectors](05_hdhdh_dropin_validation.md)**
  — direction MSE within ±0.3% of Haar on the grid; structured rotation is the
  validated 46µs production default. FINDING.
- **[06 — Finite-d P1.7 debias adopted](06_p17_debias_adopted.md)** — new metric
  tq_bias_raw_b1 (\|ratio−1\|): raw 0.357 → debiased 0.021 (−94% logit bias) at
  zero bytes. KEPT.
- **[07 — Robustness / overfit check](07_robustness_overfit_check.md)** — 6-bit
  norm & debias hold across σ∈{0.15..1.0} (≤+0.34% up to 0.5); leaderboard
  ±1.7% across seeds. FINDING.
- **[08 — Debiasing generalized to every grid rate](08_debias_all_grid_rates.md)**
  — b=2 was 10–14% logit-shrunken; finite-d debias now applied per (d,b).
  Grid logit-bias 0.209 → 0.050 (−76%) at zero bytes. New metric tq_bias_raw.
  KEPT.
- **[09 — Calibrated norm log-range [-1.2,1.2]](09_norm_logrange_calibration.md)**
  — free knob: tq_mse 0.296047 → 0.295584 (−0.16%, below original baseline) at
  same 134 b/token. First primary-metric win; better AND lighter. KEPT.
- **[10 — Lloyd-optimal norm codebook — negative](10_lloyd_norm_codebook_negative.md)**
  — +0.16% worse; norm header settled at 6-bit / range 1.2. No code change.
- **[11 — Attention-objective probe refutes the debias](11_attention_objective_refutes_debias.md)**
  — softmax-KL over a realistic key set under one serving rotation: the
  multiplicative debias (exp 06/08) HURTS attention (noise × 1/c²); reverted
  the default. New guardrail metric tq_attn_kl_b2 = 0.001245. KEPT.
- **[12 — Attention-objective rate-distortion frontier](12_attention_frontier_prod_refuted.md)**
  — scalar KL ÷~3.3 per extra bit/coord; prod b=2 is 4.6× worse attention-KL
  than scalar b=2 at 6 more bits (per-key noise ~2×); 6-bit norm confirmed
  attention-neutral. Refutes prod at matched bytes. KEPT (finding + runner).
- **[13 — Data-oblivious norm-key protection](13_norm_key_protection.md)** —
  protecting top-frac keys by stored norm (0 extra metadata) at higher
  direction bits beats uniform b=2 at same avg bytes: top1% KL −4.1%,
  top5% −12.5% (random control confirms norm-targeting real). NOT adopted
  (mixed widths break paged-fit); GPU-box candidate. KEPT (finding + runner).
- **[14 — Structured hdhdh rotation validated end-to-end](14_hdhdh_attention_validation.md)**
  — over 8 serving-rotation draws, hdhdh == Haar on attention KL/recall/
  per-key-noise (±0.6%), no per-deployment rotation risk; closes exp-04 at the
  real objective. Full serving stack validated at 134 bits. KEPT (finding).
- **[15 — K/V bit allocation on the attention objective](15_kv_bit_allocation.md)**
  — values are the output: at equal 268 bits, value-favored K1V3 output-error
  0.065 vs K2V2 0.186 vs key-favored K3V1 0.453 (~7×). Refutes "More for
  Keys" for output fidelity; keys buy distribution, values buy output. KEPT
  (finding + runner).
- **[16 — Attention fidelity is context-length stable](16_context_length_stability.md)**
  — softmax-KL flat from 128 → 8000 keys at b=1,2 (0.00157, 3-seed ratio
  0.96–1.04); per-key noise constant; hdhdh holds at 8k. Long-context serving
  validated; n_db=1500 numbers representative. KEPT (finding + runner).
- **[17 — Attention-fidelity tail risk](17_attention_tail_risk.md)** — per-query
  KL is heavy-tailed (p99 ≈ 10.6× median at b=2) and query-norm-driven
  (corr +0.94): sharp attention amplifies per-key noise. Protection helps the
  middle tail only. New guardrail tq_attn_kl_b2_p95 = 0.004156. KEPT.
- **[18 — FWHT batch-safety bugfix + dense crossover](18_fwht_batch_bugfix_crossover.md)**
  — fixed silent batch bug (shape[0]→shape[-1], perm x[...,perm]); verified
  bitwise batch==per-row. NumPy crossover quantified: dense matmul wins at all
  sizes (0.088 vs 5.0 µs/key, ~55×); FWHT O(d log d) is a GPU property. KEPT.
- **[SUMMARY — session findings & handoff](SUMMARY.md)** — the consolidated
  results, honest floor assessment, and GPU-box deployment recommendations.

## Status summary

_updated as experiments land_
- **Better (quality)**: tq_mse = 0.295584 — below the original 8-bit baseline
  (0.295923) at the lighter 134 b/token (exp 09)
- **Better (attention, real objective)**: tq_attn_kl_b2 = 0.001245 (exp 11;
  the debias default reverted because it worsened this), b=1..2 raw
  estimators retained
- **Lighter (bytes)**: tq_bytes_b2 = 134.0 (from 136, exp 03)
- **Lighter (compute)**: fwht_us = 46.5 µs (from 389, exp 02)
- **Known dead ends / decision rules**: see `.auto/prompt.md` → "Known dead
  ends" (P2.8 triplets, P2.9 joint rounding, single-Hadamard/perm rotations);
  **surrogate rule**: tq_bias_raw must NOT be optimized alone — exp 11 showed
  multiplicative debias (ratio→1) hurts softmax attention.
