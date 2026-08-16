# 04 — Empirical Codebooks for the Structured (hdhdh) Rotation — negative

**Goal:** the fast 3-round Hadamard+sign-flip rotation (hdhdh, now 46µs after
exp 02) has coordinates that are only approximately Beta-distributed. Prior
P0.3 planned an *empirical* codebook path for production ("exact-Beta for
theory, empirical for production") but never measured it. This experiment tests
whether an empirical Lloyd-Max codebook calibrated on actual hdhdh-rotated
realistic vectors beats the exact-Beta codebook on the hdhdh path.

**Method / Protocol:** for d ∈ {64, 128}, b ∈ {1, 2}: build the benchmark's
realistic dataset (uniform direction × lognormal(0, 0.35)), calibrate an
empirical symmetric b-bit Lloyd-Max codebook on hdhdh-rotated coordinates from
24 independent rotations of a held-out calibration set (robust scalar k-means,
empty-cell reseeding), then measure per-vector direction MSE under hdhdh for
exact-Beta vs empirical codebooks. 4 independent rotations averaged, same
seeds as the leaderboard grid.

**Anti-cheat check:** unchanged seeds/inputs; the calibration set is separate
from the evaluation; the comparison is apples-to-apples (same rotation, same
vectors, only the codebook differs).

**Results (direction MSE, per-vector):**

```
d=64  b=1: hdhdh/exact=0.35896   hdhdh/emp=0.35881   (+0.04% for emp)
d=64  b=2: hdhdh/exact=0.11413   hdhdh/emp=0.11471   (−0.51% for emp)
d=128 b=1: hdhdh/exact=0.36173   hdhdh/emp=0.36053   (+0.33% for emp)
d=128 b=2: hdhdh/exact=0.11570   hdhdh/emp=0.11578   (−0.07% for emp)
```

The empirical centroids converge to the exact-Beta centroids (±0.0001), e.g.
d=64/b=2: exact [−0.1875, −0.0565, 0.0565, 0.1875] vs emp [−0.1876, −0.0565,
0.0565, 0.1876].

**Findings:**
1. **No practical gain.** All four configs within ±0.5% (sampling noise); the
   empirical calibration adds zero value in this regime.
2. **Explanation:** hdhdh's coordinate marginal is statistically
   indistinguishable from the exact Beta law (the P0.1 result), so the
   exact-Beta Lloyd-Max codebook is already effectively optimal for it. The
   empirical path only matters when the true marginal deviates from Beta —
   i.e., on real KV tensors (P0.4/GPU box), not in the statistical regime.
3. Validates the codebase design choice to ship exact-Beta codebooks for the
   structured rotation.

**Tradeoffs / risks:** empirical codebooks add calibration code + a codebook
per layer for zero gain here; correctly rejected. The empirical path should be
reserved for real-data experiments.

**Verdict:** **discard / no code change** — measured negative result; the
exact-Beta codebook stands for both Haar and hdhdh in this regime.

**ASI (remember after reset):**
- Do NOT re-test empirical codebook calibration for structured rotations in the
  statistical regime — the marginal is Beta and exact-Beta LLB is optimal.
- Empiricism only pays on real KV data with non-Beta marginals (GPU box).
