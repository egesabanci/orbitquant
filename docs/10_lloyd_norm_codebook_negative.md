# 10 — Lloyd-Optimal Norm Codebook — negative

**Goal:** replace the uniform-in-log 6-bit norm grid (midpoint reconstruction,
range ±1.2) with a rate-distortion-optimal 6-bit Lloyd-Max codebook trained on
the log-norm marginal — the same "match the codebook to the marginal" step the
direction codebooks already use. Should reduce the norm component of MSE at
the same 6 bits / 134 b/token.

**Method / Protocol:** trained an asymmetric 6-bit Lloyd-Max (k-means with
quantile init, empty-cell reseeding) on log-norm samples from the benchmark's
fixed distribution over [−1.2, 1.2], then re-measured the full grid MSE with
Lloyd centroids/edges replacing the uniform grid.

**Anti-cheat check:** same bit budget, same seeds/inputs/range; only the
norm codebook (offline-calibrated, shared → 0 per-token bits) changed.

**Results (grid mean, 6-bit, range 1.2):**
```
uniform-log midpoint:  0.295584
Lloyd-opt norm codebook: 0.296069   (+0.16% WORSE)
```

**Findings:**
1. **No gain — slightly worse.** Lloyd-optimizing the norm quantizer worsens
   total MSE by +0.16%. The norm's own reconstruction error is a tiny fraction
   of the total (direction quantization dominates), and the total error goes
   through `xhat = rhat · uhat`, so improving the norm marginal alone does not
   reduce — and can slightly perturb — the measured MSE (cross-term effects).
2. Conclusion: the norm header is *not* a meaningful lever beyond the range
   calibration of exp 09. The uniform-log midpoint grid at range 1.2 is
   effectively rate-distortion fine for this regime.

**Tradeoffs / risks:** none; rejected as no-gain (and it would add a per-layer
codebook to maintain).

**Verdict:** **discard / no code change** — measured negative; confirms the
norm header is settled at NORM_BITS=6 / LOG_RANGE=1.2.

**ASI (remember after reset):**
- Do NOT pursue norm-codebook optimization (Lloyd-on-log-norm = +0.16% worse).
  The 6-bit uniform-log grid at range 1.2 is final in this regime.
- The norm contributes too little MSE to reward further engineering; real KV
  only needs the calibrated range to avoid clipping.
