# 03 — 6-bit Log Norm Header (side-info lightness)

**Goal:** reduce the honest paid-bytes/token (`tq_bytes_b2`) without material
quality loss. At b=2/d=64 the representation is 128 payload + 8 side-info bits;
the 8-bit log norm header is 100% of the side-info overhead, so it is the
lightness target. P1.3 validated 8-bit log norms but never swept to fewer bits.

**Method / Protocol:** swept `benchmark_tq.grid_mse(norm_bits=k)` — the norm is
quantized in log space over the fixed log-range [−1, 1] (midpoint
reconstruction, P1.3 format), then each vector is reconstructed as
`xhat = r_hat * u_hat` and the honest per-vector MSE against the unchanged
realistic inputs (uniform direction × lognormal(0, 0.35) magnitude) is
measured. Adopted design change: `benchmark_tq.NORM_BITS = 6`, and
`true_bits_b2()` discounts the header via P0.5 accounting (payload 128 + side 6
= 134, padding 0). No seeds, datasets, or budget formulas changed; the only
change is the representation's header width, whose quality consequence is
measured on the same inputs.

**Tradeoff sweep (norm_bits → grid mean MSE → est. true bits/token @ b=2/d=64):**

```
 8 -> 0.295923 @ 136    7 -> 0.295955 @ 135    6 -> 0.296047 @ 134
 5 -> 0.296328 @ 133    4 -> 0.297371 @ 132    3 -> 0.301425 @ 131
 2 -> 0.317537 @ 130
```

**Anti-cheat check:** identical seeds/inputs/budgets; quality consequence fully
measured on the same realistic vectors (a shorter header is not free — the
sweep quantifies exactly what it costs); bytes honestly discounted by the same
P0.5 accounting used everywhere.

**Results (before → after):**

| Metric | Before | After | Δ |
|---|---|---|---|
| tq_mse (primary) | 0.295923 | 0.296047 | +0.000124 (+0.04%) |
| tq_bytes_b2 | 136.0 | **134.0** | **−1.5%** |
| tq_bias_b1 | 0.019938 | 0.019938 | 0.0 |
| tq_var_b2 | 0.988199 | 0.988199 | 0.0 |
| fwht_us | 46.1 | 46.2 | ~0 |

**Findings:**
1. **8 bits waste ~2 bits/token in this regime.** Dropping 8 → 6 costs only
   +0.04% MSE; the norm error scales with the log-bin size (per-vector
   norm-only MSE: 8-bit ≈ 0.0002, 6-bit ≈ 0.0003) — far below the direction
   quantization noise at every grid point (0.45 / 0.14).
2. **The tradeoff is smooth and the corner is at 6 bits.** 4 bits (132 bytes,
   −1.5% more) costs +0.49% MSE — 12× the quality cost of the 6→4 step for half
   the byte saving; 2 bits collapses (+7%). 6 bits is the honest sweet spot for
   σ=0.35 log-normal norms over [−1, 1].
3. **Handling for wider dynamic ranges:** on real KV tensors the norm dynamic
   range is unknown, so a per-layer/head calibrated log-range (P1.3/P2.5)
   would make an even shorter header safe; here the fixed [−1,1] range plus
   σ=0.35 already lets 6 bits cover r∈[0.28, 3.19] with negligible error.

**Tradeoffs / risks:** 0.04% quality cost for 1.5% fewer bytes — the honest
price of lightness, accepted and documented. The byte lever at b=2/d=64 is now
exhausted in this benchmark (payload packs with zero padding; further savings
would require touching the payload rate = the quality axis).

**Verdict:** **keep** — `tq_bytes_b2` 136 → 134 (−1.5%) for +0.04% MSE
(immaterial), per the documented "lighter without material quality loss"
policy. All checks green.

**ASI (remember after reset):**
- The norm header is a smooth bytes/quality knob; 6-bit is optimal here
  (8→6 is free-ish; 6→4 costs 12× more quality per byte).
- Remaining lightness on this axis: none (payload has zero packing waste).
  Next lever: estimator fidelity (`tq_bias_b1`), then verify no honest MSE
  lever exists on the grid.
