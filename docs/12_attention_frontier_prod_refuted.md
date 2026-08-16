# 12 — Attention-Objective Rate-Distortion Frontier (and `prod` Refuted at the Real Objective)

**Goal (creative, built on exp 11's "attention is noise-limited" lens):** produce
the first rate-distortion curve measured on the REAL objective (softmax-KL over
a realistic context) instead of MSE, and answer two deployment questions nobody
has answered here: (1) what does each true-byte budget buy in attention
quality, and (2) does the paper's TurboQuant_prod (residual QJL) — promoted as
the "better estimator" on the bias/mSE surrogate — actually help attention, or
is scalar MSE the right choice at matched bytes?

**Method / Protocol (`benchmark_tq.attention_frontier`, runner
`baseline/run_frontier.py`):** one shared serving rotation P encodes a
realistic key cache; realistic queries score the stored codes in the rotated
domain; softmax-KL and recall@5 vs full precision, per config. Configs: scalar
b∈{1,2,3,4} (bits = b·d + 6-bit norm), scalar b=2 with 8-bit norm, and prod
b=2 (1-bit MSE base + m=64 QJL sign bits + 6-bit key norm + 6-bit residual
norm = 140 bits; residual norm honestly 6-bit-quantized). Deterministic.

**Anti-cheat check:** nothing fitted; all configs through the same serving
probe; byte accounting follows P0.5; the `prod` result was independently
verified as a distributional property (per-key score-error std) and matches
the per-pair variance sitting at its own Thm-2 bound.

**Results (d=64):**

```
config                  bits/tok  bytes/tok      KL     recall@5
scalar b=1                   70       8.75   0.00483     0.405
scalar b=2                  134      16.75   0.00157     0.685
scalar b=2, 8-bit norm      136      17.00   0.00157     0.690
prod b=2 (1bit+QJL m=64)    140      17.50   0.00724     0.410
scalar b=3                  198      24.75   0.00044     0.795
scalar b=4                  262      32.75   0.00012     0.870
```

Per-key score-error std (verified): scalar b=2 ≈ 0.036–0.048; prod b=2 ≈
0.082–0.097 (≈2× worse); prod's variance sits at its own Theorem-2 bound,
which is itself ~3.5× looser than scalar's actual attention noise.

**Findings:**
1. **Attention-KL is strongly superlinear in rate:** each extra bit/coord cuts
   KL ~3.1–3.6× (b1→b2 3.07×, b2→b3 3.5×, b3→b4 3.6×). The scalar rate curve
   on the REAL objective is smooth and monotone — for a deployment budget this
   is the curve to schedule against (e.g., b=3 costs 1.47× bytes for 3.5× less
   attention distortion).
2. **The MSE surrogate ranks scalar rates correctly** (b1 0.449 > b2 0.143 >
   b3 ~0.034 > b4 ~0.009, matching KL order) — no overfit on RATE selection.
3. **TurboQuant_prod b=2 is 4.6× worse on attention-KL than scalar b=2 at 6
   MORE bits** (recall@5 error 1.87×). Mechanism: at matched bytes the
   residual-QJL path puts half its payload into a coarse 1-bit base plus a
   high-variance per-key sketch; for noise-limited softmax attention that
   variance is what matters, and it is ~2× scalar's. The paper's product
   "variance at the Theorem-2 bound" only means it is optimal *for its own
   estimator*; that bound is far looser than scalar MSE at the same rate.
4. **Norm-header design is attention-insensitive** (8-bit vs 6-bit: KL 0.00157
   vs 0.00157, recall 0.690 vs 0.685) — exp-03/09's 6-bit choice is right on
   the real objective (saves 2 bits/tok for ~0).

**Implications / verdict:** scalar TurboQuant_mse is the attention-correct
default at every rate in this regime; `prod` and the debias (exp 11) are
surrogate-optimal only and should not be default. No default change this
iteration (scalar b=2 stays); the frontier and per-key noise numbers are the
deployment artifact + a cautionary, quantitative refutation of "prod is better"
at matched bytes. **keep as finding + runner.**

**ASI (remember after reset):**
- The attention-objective frontier table above is the deployment reference.
- prod/QJL residual at matched bytes ≈ 2× scalar's per-key score noise (sits
  at its own, looser, bound). Never compare "estimator variance vs its own
  theorem bound" across estimators — compare the actual per-key score-noise
  at matched bytes (here: scalar b=2 std 0.04 vs prod std 0.09).
- Runner: `python -m baseline.run_frontier`.
