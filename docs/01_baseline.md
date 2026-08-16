# 01 — Baseline Measurement

**Goal:** establish the honest statistical numbers of the current TurboQuant
baseline (exact-Beta Lloyd-Max scalar codebooks, Haar rotation, 8-bit log norm
header) so every later experiment is compared against a stable, reproducible
reference.

**Metrics (primary first; ↓ = lower better):**

| Metric | Baseline (this doc) | Meaning |
|---|---|---|
| `tq_mse` (primary) | **0.295923** | mean per-vector MSE over d∈{64,128} × b∈{1,2}, realistic log-normal norms, explicit 8-bit log norm header |
| `tq_bytes_b2` | **136.0** bits/token | honest true bits/token at b=2/d=64 (P0.5 accounting: 128 payload + 8 side, 0 padding) |
| `tq_bias_b1` | **0.019938** | \|inner-product bias ratio − 2/π\| at b=1 (exact-Beta ratio ≈ 0.657 vs 2/π ≈ 0.637) |
| `tq_var_b2` | **0.988199** | product-estimator variance / Theorem-2 bound (measured base MSE, fair at any d) |
| `fwht_us` | **≈ 389** µs (372–398) | best-of-40 apply time of 3-round FWHT structured rotation at d=128 |

**Protocol:** fixed seed `20260816`; dataset of 1200 vectors per config, direction
uniform on the sphere, magnitude log-normal (σ=0.35), so the norm header is
genuinely required. Quality grid averages independent Haar rotations per config
(N_ROT=4). Rates are fixed (payload b·d + 8-bit log norm header); nothing can
exceed its stated budget. Everything except `fwht_us` is fully seeded →
deterministic (zero noise); the benchmark harness medians `fwht_us` over 3
runs.

**Per-config grid (tq_mse components):**

```
d=64  b=1  mse=0.44899
d=64  b=2  mse=0.14263
d=128 b=1  mse=0.44744
d=128 b=2  mse=0.14462
mean      0.29592
```

**Interpretation / headroom assessment:**
- `tq_mse` ≈ 0.296 is dominated by b=1 (≈0.45 vs ≈0.14 at b=2). Scalar
  exact-Beta Lloyd-Max is near the information-theoretic optimum here
  (TurboQuant Thm 1 + prior negative results on triplet/joint codecs), so MSE
  headroom is small and honest.
- `tq_bytes_b2 = 136.0`: the 8-bit log norm header is **100% of the overhead**
  over raw payload (128). This is the cleanest "lighter" target — compress the
  norm side-info format, measure the norm-error vs bits tradeoff.
- `tq_bias_b1`: small but real (0.0199) correction headroom on the b=1 MSE
  estimator.
- `tq_var_b2` is already at the Theorem-2 bound (0.988) — the product estimator
  is implemented faithfully; little headroom.
- `fwht_us` ≈ 389 µs is the pure-Python Python-loop butterfly (slow vs BLAS
  dense, known from P0.1 report) — vectorizing it is a real, provably-
  equivalent compute-win target.

**Verdict:** baseline recorded; nothing to keep/discard (reference only).
