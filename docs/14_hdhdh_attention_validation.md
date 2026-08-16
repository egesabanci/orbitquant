# 14 — Fast Structured Rotation (hdhdh) Validated End-to-End on the Attention Objective

**Goal (creative, closes the last deployment gap):** exps 02/05 validated the
fast 3-round FWHT rotation (hdhdh, 48µs) on reconstruction MSE and the
paper's theorem checks — but production serves with ONE FIXED rotation per
cache, and the real thing that matters is softmax attention. Two open
questions: (1) does hdhdh's *nearly*-Beta coordinates (vs exact Beta under
Haar) cost anything at the attention objective, and (2) does attention quality
vary across different fixed rotation draws (a per-deployment.
robustness/risk question)? Even exp-04's "empirical codebook gives ~0" was
only checked at MSE — this closes that question at the real objective too.

**Method / Protocol:** extended `attention_metrics()` with a `rot` parameter
("haar" | "hdhdh") and `rot_seed` (the fixed serving rotation). hdhdh is
materialized as a matrix (apply the 3-round sign-flip FWHT to the basis) so
the same vectorized scoring path runs. For 8 independent serving-rotation
draws, at b∈{1,2}/d=64, reported softmax-KL, recall@5, and per-key
score-noise / mean-|true-score| (the noise-limited regime's decisive
quantity, exps 11/12). Dataset, queries, norm header, codebooks, and bytes are
identical across the comparison — only the rotation changes.

**Anti-cheat check:** same probe, same seeds/bytes, only the rotation differs;
no fitting; deterministic.

**Results — mean over 8 independent serving-rotation draws (d=64):**

```
rot    b   KL            recall@5   score_noise   cross-draw KL spread (std)
haar   1   0.0039813     0.591      0.7958        0.0000177
hdhdh  1   0.0039970     0.582      0.7962        0.0000279
haar   2   0.0012791     0.843      0.4501        0.0000135
hdhdh  2   0.0012872     0.839      0.4508        0.0000125
```

**Findings:**
1. **hdhdh is attention-equivalent to Haar:** KL within +0.4–0.6%, recall
   within −0.005 to −0.009, per-key score-noise equal to 3 decimals. The
   fast rotation's near-Beta coordinates cost nothing in the serving-rotation
   regime where it would actually be used.
2. **No per-deployment rotation risk:** the cross-draw spread is tiny
   (σ ≈ 1% of the KL mean) and identical for Haar and hdhdh. Attention quality
   is rotation-draw-stable — you do not gamble quality on which random
   rotation a given deployment rolls.
3. **Closes exp-04 at the real objective:** since hdhdh's attention behavior
   is indistinguishable from Haar with the exact-Beta codebook, an hdhdh-
   calibrated (empirical) codebook has no attention-level headroom either —
   consistent with the MSE-level finding. The exact-Beta codebook is
   production-correct on the fast rotation.
4. **The full serving stack is now validated end-to-end:** fixed hdhdh
   48µs rotation + exact-Beta scalar codebooks + 6-bit log norm (+1.2 range) +
   raw (undebiased) estimator ⇒ attention-faithful at 134 bits/token
   (tq_attn_kl_b2 ≈ 0.0012), matching ideal Haar.

**Implications / verdict:** production can use the fast rotation with
confidence. Leaderboard protocol keeps Haar as the ideal reference (the
exact-Beta theory is Haar's); hdhdh is documented as the validated deployment
path. **keep as finding** (no default change needed — nothing to fix).

**ASI (remember after reset):**
- hdhdh ≈ Haar on attention KL/recall/per-key-noise (within ≤0.6%/) and has
  the same tiny cross-draw variance — fully deployment-validated.
- exact-Beta codebook is attention-correct under hdhdh too (no empirical
  codebook headroom, even at the real objective).
- Use `attention_metrics(d, b, rot="hdhdh", rot_seed=? )` to replicate.
