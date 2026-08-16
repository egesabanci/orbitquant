# Autoresearch: Make TurboQuant Better and Lighter (statistical regime)

## Objective

Improve the **original TurboQuant** vector-quantization pipeline on two honest
axes, within OrbitQuant's pure-NumPy / no-model / no-GPU statistical regime:

1. **Better** — lower per-vector reconstruction MSE and / or higher
   inner-product estimator fidelity (bias ~ 2/pi, product-estimator variance
   vs the Theorem-2 bound), at fixed honest bit rates.
2. **Lighter** — fewer true paid bytes/token (P0.5 accounting: payload +
   side-info + packing) and lower compute (vectorized FWHT / structured
   rotation), *without* regressing quality.

The theory says exact-Beta Lloyd-Max scalar quantization on Beta coordinates is
already near-optimal (TurboQuant Thm 1), so `tq_mse` headroom is small and
honest. The genuine, reachable headroom is on `tq_bytes_b2` (side-info
formats), `tq_bias_b1` / `tq_var_b2` (estimator fidelity), and `fwht_us`
(compute). Several related negative results are ALREADY established (see
"Known dead ends") — do not re-attempt them.

## Metrics

Primary metric from `init_experiment`: **`tq_mse`** (lower is better) — mean
per-vector reconstruction MSE over the fixed honest grid
d∈{64,128} × b∈{1,2}, on realistic log-normal-norm vectors with an explicit
8-bit log norm header.

**Standing loop policy (important deviation, matches the user's dual goal):**
a change may be kept if it improves ANY metric below while `tq_mse` does not
regress (beyond noise) — because the session's second declared goal
("lightweight") maps to `tq_bytes_b2`/`fwht_us`, not to the primary. Otherwise
follow the normal keep/discard rules. Never keep a change that regresses
`tq_mse` materially unless another axis improves and you document the tradeoff
(and it is usually the wrong call).

| Metric | Unit | Better | Meaning |
|---|---|---|---|
| `tq_mse` (primary) | unitless | lower | mean per-vector MSE over grid (quality) |
| `tq_bytes_b2` | bits/token | lower | honest paid bits/token at b=2/d=64 (lightness) |
| `tq_bias_b1` | unitless | lower | \|inner-product bias ratio − 2/π\| at b=1 (logit fidelity) |
| `tq_bias_raw_b1` | unitless | lower | \|inner-product bias ratio − 1.0\| (TRUE estimator unbiasedness; SURROGATE ONLY — exp 11: optimizing it misleads, do NOT act on alone) |
| `tq_bias_raw` | unitless | lower | grid mean \|ratio−1\| (SURROGATE ONLY — same caveat) |
| `tq_attn_kl_b2` | unitless | lower | softmax-KL over realistic key set under one serving rotation, b=2/d=64 — THE REAL OBJECTIVE (exp 11) |
| `tq_var_b2` | ratio | lower | product-estimator variance / Thm-2 bound (logit fidelity) |
| `fwht_us` | µs | lower | best-of-N apply time of 3-round FWHT at d=128 (compute) |

Current best (commit 4fb137b): tq_mse 0.295584 (BELOW original 0.295923),
tq_bytes_b2 134, tq_bias_b1 0.019938, tq_bias_raw_b1 0.357 (raw/undebiased:
the debias default was REVERTED exp 11), tq_bias_raw 0.209 (raw), tq_attn_kl_b2
0.001245, tq_var_b2 0.988199, fwht_us ~47. IMPORTANT: the finite-d debias
(exp 06/08) was refuted at the real attention objective by exp 11 (it amplifies
per-key logit noise by 1/c^2 and softmax is noise-limited) and withdrawn.
Do not re-optimize tq_bias_raw alone.

`tq_mse`, `tq_bytes_b2`, `tq_bias_b1`, `tq_var_b2` are fully seeded →
deterministic (zero noise); `fwht_us` has ~2% timing noise (measure.sh medians
it over 3 runs).

## How to Run

- Bench: `bash .auto/measure.sh` → prints `METRIC name=value` lines (primary
  first). Use `run_experiment` (times it, captures output).
- Checks: `bash .auto/checks.sh` → py_compile all modules, benchmark
  invariants, and the six TurboQuant theorem checks at nrot=2000 (deterministic,
  exits nonzero on failure). This runs automatically after each passing bench.
- Always `log_experiment` after every `run_experiment`.

## Files in Scope

You may modify anything under `baseline/` **except the benchmark itself if the
change would game the numbers** (see Anti-cheat contract). Most likely levers:

- `baseline/codebooks.py` — exact-Beta Lloyd-Max solver, quantization primitives
  (better/faster codebooks, codebook caching, quantization methods).
- `baseline/benchmark_tq.py` — the benchmark. DO NOT weaken it (change seeds,
  inputs, budgets, or metric formulas) to produce fake gains. LEGITIMATE edits:
  (a) exposing new honest diagnostics; (b) reflecting an ADOPTED design change
  that is itself being measured (e.g. exp 03 changed the representation's norm
  header 8→6 bits and exp 06 added the honest tq_bias_raw_b1 metric + adopted
  finite-d debias as the measured scoring path). In both cases the quality
  consequence was measured on the same inputs/seeds — that is the difference
  between reflecting design and gaming measurement. Document any such edit.
- `baseline/rotations.py` — the FWHT butterfly / structured rotations (compute
  lightness target: vectorize the pure-Python butterfly WITHOUT changing its
  output — must stay bitwise-equivalent to the reference for the same input).
- `baseline/p05_accounting.py` — honest byte accounting (lightness measure).
- `baseline/corrections.py`, `baseline/protocol.py`, other proposal modules —
  as building blocks for new encoders/estimators.

Autoresearch infrastructure (treat as yours, do NOT gate games through it):
- `.auto/measure.sh`, `.auto/checks.sh`, `.auto/prompt.md`, `.auto/ideas.md`,
  `.auto/log.jsonl`.
- `docs/` — REQUIRED output: one clean, single-responsibility markdown doc per
  experiment (see Docs convention).

## Off Limits

- `tests/`, `baseline/checks.py`, `.auto/checks.sh`: must keep passing. Do not
  edit them to force acceptance (that is cheating).
- The benchmark's seeds, datasets, budgets, and metric definitions must not be
  changed to game results.
- Do not add third-party dependencies (pure NumPy only, as the project
  mandates).
- Do not touch `papers/`, `proposals.md`, `notes/`, `reports/` (historical
  records of the prior effort).

## Anti-cheat contract (read before every experiment)

1. Inputs are realistic vectors: uniform direction, log-normal magnitude, so
   norm side information is required — dropping it must measurably lose.
2. Rates on the quality grid are fixed (payload b·d + explicit 8-bit log norm
   header, mirrored by P0.5 honest accounting).
3. Results average over d∈{64,128}, b∈{1,2} — one-config overfitting is not
   rewarded.
4. Do not change seeds/inputs/budgets to make your change win. A real win must
   show on the unmodified benchmark.
5. If your change is a strict algebraic rewrite (e.g., vectorized FWHT), prove
   equivalence: same outputs as the reference on random inputs (selfcheck
   already checks `fwht(fwht(x)) == d·x` and hdhdh orthogonality; add a
   direct numeric comparison in your doc).

## Docs convention (the user's explicit requirement)

Write findings to `docs/` as **one markdown file per experiment**, cleanly
separated by single responsibility (SoC), following `docs/_template.md`:

- `docs/00_index.md` — running table of contents (append every experiment).
- `docs/01_baseline.md` — the baseline measurement (first doc).
- `docs/0X_<short-name>.md` — each subsequent experiment: Goal, Method /
  Protocol, Results (table with before/after per metric), Findings, Tradeoffs,
  Verdict (keep/discard/crash), ASI-what-to-remember.
- Update `.auto/ideas.md` with any deferred-but-promising ideas.
- APPEND/UPDATE `.auto/prompt.md` "What's Been Tried" after each experiment so
  a resuming agent has full context.
- Keep each doc focused on ONE idea so it is reviewable in isolation.

## What's Been Tried

- **EXP 02 (kept): Vectorized FWHT** (`baseline/rotations.py::fwht`). Replaced
  the per-block Python loop with strided `reshape(-1,2h)` + slice add/sub per
  level. Bitwise-identical output (verified d∈{4..512}), fwht_us 389→46 (8.4×),
  all other metrics unchanged. The 14µs/level floor is numpy-call overhead;
  further numpy-level micro-optimization has diminishing returns. See
  `docs/02_fwht_vectorization.md`.
- **EXP 03 (kept): 6-bit log norm header** (`baseline/benchmark_tq.py`
  NORM_BITS 8→6). tq_bytes_b2 136→134 (1.5%) at tq_mse 0.295923→0.296047
  (+0.04%). Sweep: 8→6 ≈ free; 6→4 costs +0.49% MSE (12× per-byte); the
  8-bit header wasted ~2 b/token here. Byte levers now exhausted at b=2/d=64
  (zero packing waste). See `docs/03_norm_header_6bit.md`.
- **EXP 04 (negative, no code): empirical codebooks for hdhdh give ~0 gain** —
  empir LLB converges to exact-Beta (±0.0001), ±0.5% noise. exact-Beta is
  optimal for the structured rotation in this regime. Empiricism only pays on
  real-KV marginals (GPU box). Also confirmed P1.8 norm-preserving WORSENS
  grid MSE (0.296→0.322) — no MSE lever. See `docs/04_...`.
- **EXP 05 (finding, no code): hdhdh ≈ Haar on realistic grid (±0.3%)** — the
  fast 46µs structured rotation is a validated production drop-in. See
  `docs/05_...`.
- **EXP 06 (kept): finite-d P1.7 debias adopted as default scoring path** +
  new metric `tq_bias_raw_b1` (\|ratio−1\|, true estimator fidelity).
  Raw 0.357 → debiased 0.021 (−94% logit bias) at zero bytes (global scalar).
  See `docs/06_p17_debias_adopted.md`.
- **EXP 07 (finding, no code): robustness/overfit check** — 6-bit norm +
  debias hold across σ∈{0.15..1.0} (≤+0.34% up to σ=0.5; +1% at 0.75);
  wide-range degradation is range-clipping → fix = calibrated per-layer
  ranges, not more bits. Leaderboard ±1.7% across dataset seeds. See
  `docs/07_robustness_overfit_check.md`.
- **EXP 08 (kept): finite-d P1.7 debias generalized to every grid rate** —
  b=2 raw was ~0.86-0.90 (10-14% logit shrink; 2/π is only a b=1
  asymptotic); debias now per-(d,b); new metric tq_bias_raw (grid mean
  \|ratio−1\|): 0.209 → 0.050 (−76%) at zero bytes. See
  `docs/08_debias_all_grid_rates.md`.
- **EXP 09 (kept): norm log-range 1.0 → 1.2** (LOG_RANGE in benchmark). -
  tq_mse 0.296047 → 0.295584 (−0.16%, below the original 8-bit baseline
  0.295923) at the same 134 b/token — first primary win; better AND lighter.
  Optimum at 1.2 (1.5 coarsens bulk bins). See `docs/09_...`.
- **EXP 10 (negative, no code): Lloyd-optimal 6-bit norm codebook** is +0.16%
  WORSE than uniform-log midpoint — norm contribution to MSE is dominated by
  direction error; header settled at NORM_BITS=6/LOG_RANGE=1.2. Don't
  re-attempt. See `docs/10_...`.
- **EXP 11 (kept): attention-objective probe refutes the debias.** Added
  `attention_metrics()` / `tq_attn_kl_b2` (softmax-KL over a realistic
  key+query set under ONE shared serving rotation). Found the multiplicative
  debias (exp 06/08) HURTS attention: it multiplies per-key logit noise by
  1/c^2 (x2.4 at b=1) and softmax is noise-limited; KL worse at every rate
  even with a realistically-calibrated factor (b=1: 0.0044->0.0064, b=2:
  0.0014->0.0015). REVERTED DEFAULT_DEBIAS to False. Surrogates back up
  (tq_bias_raw 0.209, b1 0.357) -- documented as rejected-proxy cost; real
  objective improved; primary unchanged. NEVER optimize tq_bias_raw alone
  again. See `docs/11_attention_objective_refutes_debias.md`.
- **EXP 12 (kept, finding + runner): attention-objective rate-distortion
  frontier** (`attention_frontier`, `baseline/run_frontier.py`). Scalar KL
  ÷3.1-3.6x per extra bit/coord (b1 0.0048@70b → b2 0.0016@134b → b3
  0.00044@198b → b4 0.00012@262b); MSE surrogate ranks rates correctly. 8-bit
  vs 6-bit norm: KL identical (validates exp-03/09 on the real objective).
  REFUTES prod at matched bytes: prod b=2 is 4.61x worse attention-KL
  (0.00724) than scalar b=2 at 6 more bits; per-key score-error std ~0.09 vs
  ~0.04; prod sits at its own (3.5x looser) Thm-2 bound. Lesson: never judge
  an estimator against its own theorem bound; compare per-key score-noise at
  matched bytes. Scalar MSE is the attention-correct default. See
  `docs/12_attention_frontier_prod_refuted.md`.
- **EXP 13 (kept, finding + runner): data-oblivious norm-key protection**
  (`attention_protection`, `baseline/run_protection.py`). Protect top-`frac`
  keys by stored norm (decoder infers pool from shared rule → 0 extra
  metadata) with higher direction bits: same avg bytes (134.6 vs 134): top1%
  KL −4.1%, recall5 0.685→0.700; top5% −12.5%; random control (0.001558 vs
  uniform 0.001571, topnorm 0.001507) proves norm-targeting real; robust 3
  seeds × n_q=150. NOT adopted as default (mixed per-key widths break P0.5
  paged-fit/fused kernels — deployability cost beats −4% KL in a lighter
  session); GPU-box candidate (OSCAR/RotateKV-style). Norm-side protection:
  ~0 headroom (norm error negligible, exp 12). See
  `docs/13_norm_key_protection.md`.
- **EXP 14 (kept, finding): hdhdh validated end-to-end on the attention
  objective.** Extended `attention_metrics(d,b,rot=...,rot_seed=...)` to
  support the fixed fast rotation + added per-key `score_noise`.
  Over 8 serving-rotation draws (d=64, b∈{1,2}): hdhdh KL +0.4–0.6% of Haar,
  recall within 0.005–0.009, noise equal 3 decimals; cross-draw spread
  σ≈1% for BOTH → no per-deployment rotation risk. Closes exp-04 at the real
  objective (no empirical-codebook headroom on the fast rotation). Full
  serving stack (hdhdh + exact-Beta + 6-bit norm + raw estimator @134 bits)
  validated. Leaderboard keeps Haar as the ideal reference; hdhdh is the
  documented production path. See `docs/14_hdhdh_attention_validation.md`.
- **EXP 15 (kept, finding + runner): K/V bit allocation on the KV attention
  objective** (`kv_attention_error(b_k,b_v,est_p=...)`,
  `baseline/run_kv_alloc.py`). Values are consumed as out=Σp_i·v_i → value
  quant error enters the output linearly; key error only reshapes p. At 268
  bits (3 seeds × n_q=80): K1V3 rel_out_err 0.065 vs K2V2 0.186 vs K3V1
  0.453 (~2.9× / ~7×). V-only: 0.046/0.177/0.481 at b_v=3/2/1; key KL
  0.0004/0.0014/0.0043 at b_k=3/2/1. Mechanism: ratio metric ≈ value-codec
  relative MSE (concentration Σp² cancels). REFUTES "More for Keys" for
  output fidelity; reframes P1.2 as a two-objective tradeoff (keys →
  distribution, values → output). No default change (benchmark is key-side);
  GPU-box re-validation on real value marginals. See
  `docs/15_kv_bit_allocation.md`.
- **EXP 16 (kept, finding + runner): attention fidelity is context-length
  stable** (`attention_context_scale`, `baseline/run_context_scale.py`).
  Swept n_db 128→8000 with the same rotation/queries/pipeline: b=2 KL
  0.00153→0.00157 (flat; 3-seed 8000/1500 ratios 0.96/1.04/1.00), b=1 ~0.0049
  flat; score_noise 0.45 and p95 0.91 constant; recall@5 drops (0.90→0.82)
  only because top-k is harder with more candidates, not quantization.
  hdhdh @8k = 0.001559 ≈ Haar. Long-context serving validated; n_db=1500
  headline numbers representative to ≥8k. See
  `docs/16_context_length_stability.md`.
- **EXP 17 (kept, finding + new guardrail): attention-fidelity tail risk**
  (`attention_kl_tail`, metric `tq_attn_kl_b2_p95`). Per-query KL is heavily
  right-skewed: b=2 mean 0.00174 / p50 0.00117 / p95 0.00416 / p99 0.0132
  (p99 = 10.6× median), and driven by QUERY NORM (corr +0.94–0.95) — sharp
  attention amplifies per-key score noise. Protection (exp 13) cuts p95
  −7.7% vs mean −4.0% but p99 only −4.2% (middle-tail benefit only).
  Mean metrics understate risk → new permanent guardrail
  tq_attn_kl_b2_p95 = 0.004156. Real-model query-norm check = GPU box. See
  `docs/17_attention_tail_risk.md`.

## Known dead ends (from the prior effort — do NOT re-attempt)

- **OCTOPUS-style triplet direction+norm codec** (P2.8): loses to scalar
  Lloyd-Max at every matched budget after Haar rotation (coordinates are
  already near-independent + exact-Beta marginal). Not competitive in this
  regime.
- **Joint rounding for small-block codecs** (P2.9): ~0-6% gain, only repairs a
  codec that starts worse.
- **Single-round Hadamard / random permutation** as rotations: fail the
  distribution/independence gates; break TurboQuant's assumptions.
- **Pure-Python FWHT is slow** (P0.1): the established open target is to
  VECTORIZE it so the O(d log d) structured rotation is actually fast — this is
  a legitimate, provably-equivalent change.
- **Degenerate Lloyd-Max codebooks**: fixed already (tests guard d=128/b=4);
  keep codebooks exact.
- **P2.5 covariance rotations / real-KV stuff**: needs real KV data (GPU box,
  P0.4) — out of scope for this session.

## Good starting vectors for the loop

1. **Side-info lightness** (`tq_bytes_b2`, no quality loss): the 8-bit log norm
   header is 100% of the overhead at b=2. Measure the norm-error vs bits
   tradeoff (grid_mse's norm_bits parameter). If a 4-6 bit log norm keeps
   tq_mse identical, the honest true-bits/token drops 136 → 132-134 without
   regressing quality. This is the cleanest "lighter" win.
2. **FWHT vectorization** (`fwht_us`): the butterfly in `rotations.fwht` is a
   per-step Python loop. Vectorize each `h`-level with strided slice arithmetic
   and prove bitwise-equivalent output. Expected: several-× faster, zero
   quality change. Keep the QJL/rotations selfcheck green.
3. **Estimator fidelity** (`tq_bias_b1`, `tq_var_b2`): exact-Beta b=1 bias is
   0.657 vs 2/π≈0.637 (error 0.0199). Test tighter correction factors
   (empirical vs analytic) or estimator variants; product-estimator variance is
   already at the bound (0.988), so gains here are small — but any real one
   counts.
4. **Codebook precision/robustness** (`tq_mse`): tiny honest headroom; only
   pursue if there is a real mathematical or numerical improvement (e.g.,
   cheaper exact-codebook evaluation, better grid in the Lloyd-Max solver).
5. **Rate split under fixed budget**: within the 136-bit budget, is b=2 scalar
   always the best, or does (b=1 + informative residual/other) ever beat it? Use
   the existing proposal modules (p24_varsize, p22_structured_qjl) as
   references — but only if an honest measurable win appears.

Do not force gains where the theory says none exist; a well-measured negative
result is a valid experiment (document it, it counts as a finding).
