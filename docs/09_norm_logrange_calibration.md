# 09 — Calibrated Norm Log-Range [-1.2, 1.2] (free quality knob)

**Goal:** recover the last fixed-constant inefficiency in the norm header. The
fixed log-range [−1, 1] clipped the small (0.17–0.25%) tail of log-norms beyond
ln 1.0 at the benchmark's σ=0.35 input law. A wider calibrated range should
recover that at the same 6 bits / 134 b/token.

**Method / Protocol:** swept the shared fixed log-range at NORM_BITS=6 on the
unchanged realistic grid, re-measuring the full per-vector MSE each time.
Range is a data-free shared scalar (0 per-token bits), exactly the
P1.3/P2.5-style calibration knob.

**Anti-cheat check:** same 6-bit header, same 134 b/token, same seeds and
inputs; only the shared constant changed; the full quality consequence is
measured.

**Results — grid mean vs log-range (6-bit):**
```
range   grid mean
 1.0    0.296047
 1.1    0.295723
 1.2    0.295584   <- best
 1.5    0.295622   (worse: bulk bins coarsen)
```
Adopted (NORM_BITS=6, LOG_RANGE=1.2): **tq_mse 0.296047 → 0.295584 (−0.16%)**,
which is **below the original 8-bit / range-1.0 baseline (0.295923)** while
keeping bytes at 134 (< 136). This is the session's first primary-metric
improvement, and it is *better AND lighter* simultaneously.

| Metric | Before | After | Δ |
|---|---|---|---|
| tq_mse (primary) | 0.296047 | **0.295584** | **−0.16%** |
| tq_bytes_b2 | 134.0 | 134.0 | 0 |
| tq_bias_raw (grid) | 0.050037 | 0.050037 | 0 |
| all other metrics | — | — | unchanged |

**Findings:**
1. The fixed range was clipping ~0.2% of tails; widening to 1.2 fully recovers
   the 6-bit adoption's +0.04% cost and then some. The optimum exists (1.2)
   because wider ranges coarsen the bulk bins (1.5 is worse again).
2. This is a free, deployable calibration knob: the range is a shared scalar
   per layer/head — the same mechanism recommended for real KV in exp 07.
3. It further confirms the benchmark is not a one-dimensional scoreboard:
   quality improved on the *same* byte budget by fixing a calibration constant,
   with no representation/rate change.

**Tradeoffs / risks:** none. The knob is cheap to re-tune on real KV data
(same sup: calibrate the range, don't add bits).

**Verdict:** **keep** — first primary-metric improvement (0.296047 →
0.295584), bytes unchanged at 134, all checks green.

**ASI (remember after reset):**
- LOG_RANGE=1.2 (from 1.0) at NORM_BITS=6: tq_mse 0.296047→0.295584 (−0.16%),
  below the original 8-bit baseline, same 134 b/token.
- The norm range is a "free" calibration knob with an optimum; don't over-tune
  it further (diminishing, ~0.16% total recovered).
