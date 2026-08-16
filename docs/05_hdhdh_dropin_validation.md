# 05 — Structured (hdhdh) Rotation is a Drop-in for Haar on Realistic Vectors

**Goal:** establish whether the fast 3-round Hadamard+sign-flip rotation
(hdhdh, 46µs after exp 02) can replace dense Haar rotation on the benchmark's
*realistic* input pipeline with zero quality loss — i.e., whether the
compute-light path is also a quality-identical production path.

**Method / Protocol:** using the benchmark's realistic vectors (uniform
direction × lognormal(0, 0.35) magnitude) and the exact-Beta Lloyd-Max
codebook, measure per-vector reconstruction MSE under Haar vs hdhdh for
d ∈ {64, 128}, b ∈ {1, 2}. 4 independent rotations per config, the same seeds
as the leaderboard grid. Numbers below are the direction component (norm
reconstruction is independent of rotation choice).

**Anti-cheat check:** unchanged seeds/inputs; both rotations apply to the same
vectors through the same codebook; only the transform (dense vs structured)
differs.

**Results (direction MSE, per-vector):**

```
        Haar/exact     hdhdh/exact
d=64  b=1  0.35944       0.35896      (−0.13%)
d=64  b=2  0.11395       0.11413      (+0.16%)
d=128 b=1  0.36070       0.36173      (+0.29%)
d=128 b=2  0.11584       0.11570      (−0.12%)
```

All within ±0.3% (sampling noise). Combined with the P0.1 six-check protocol
(fixed adversarial e1, β-KS, covariance, product variance vs the Theorem-2
bound) — where hdhdh was statistically indistinguishable from Haar — this
extends the equivalence to the realistic-vector benchmark directly.

**Findings:**
1. **hdhdh is quality-identical to Haar on the realistic grid** (≤0.3%).
   TurboQuant's structural assumptions (near-independent Beta-distributed
   coordinates) survive the structured transform, so the fast rotation can be
   the production default at 46µs (exp 02) without touching quality.
2. This closes the loop opened by P0.1's cap run: the only remaining
   difference is per-call compute (BLAS dense ≈ 1.9µs at d=128 beats the
   NumPy FWHT for a single vector at small d), which is a GPU/vector-hardware
   question, not a quality question.
3. Consequence: the "better and lighter" lever set is now — quality fixed on
   both rotation paths; the structured rotation + exact-Beta codebooks are
   jointly validated as the deployable configuration.

**Tradeoffs / risks:** choosing hdhdh as default trades ~0.3% sampling noise
for 8.4× less rotation compute on this metric; on GPU the FWHT butterfly is
the right choice at any head dim. No byte overhead difference (rotation
parameters are O(d) sign flips, not stored per token).

**Verdict:** **keep as finding** (no leaderboard code change — the benchmark
leaderboard intentionally still uses Haar as the reference protocol; this
documents hdhdh as the validated production alternative at 46µs).

**ASI (remember after reset):**
- hdhdh ≈ Haar on realistic vectors (≤0.3%); production can default to hdhdh.
- exp 02 (vectorized FWHT, 46µs) + exp 05 (quality-equivalence) together make
  the structured rotation the honest "lighter, same-quality" recommendation.
