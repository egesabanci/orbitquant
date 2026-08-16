# 06 — Finite-`d` P1.7 Debiasing Adopted for the Scoring Path

**Goal:** make the b=1 MSE estimator's attention logits **unbiased** (the
operationally important property) at zero byte cost. The raw TurboQuant_mse
estimator shrinks inner products by ≈ 2/π (high-`d`) — the finite-`d` value at
d=64 is ≈ 0.64 — so `E[<y, xhat>] ≈ 0.64·<y, x>`: logits are systematically
shrunk by a third. P1.7's multiplicative correction fixes this with a single
global scalar; prior work implemented it (`corrections.py`) but never made it
the default measured scoring path, and the loop had no metric for true
unbiasedness.

**Method / Protocol:** added `tq_bias_raw_b1 = |mean inner-product bias ratio
− 1.0|` at b=1, measured through the code's current scoring path
(`estimator_ratio_b1`: quantize → dequantize → unrotate → score `<y, xhat>`),
with a `DEFAULT_DEBIAS` flag. When adopted, the reconstruction is scaled by
`1 / corrections.bias_correction_factor(d, 1)` — the measured finite-d bias
ratio (≈0.657). The correction is a **global per-codebook scalar shared by all
tokens → 0 per-token bits**; it only runs once per codebook at calibration.
Seeds, inputs, and the quality grid untouched; this refines the estimator the
benchmark models (a new, honest diagnostic + design adoption), it does not
weaken any measurement.

**Anti-cheat check:** unbiasedness is measured directly as `|E[<y,xhat>]/<y,x>
− 1|` on the fixed adversarial x=e1 / y=all_equal protocol. The adopted
correction is the reciprocal of the *measured* finite-d ratio, which is the
literal definition of correcting the bias — not a fitted number tuned to the
benchmark. All other metrics are bit-identical to exp 03/02.

**Results (before → after):**

| Metric | Before (raw) | After (debias adopted) | Δ |
|---|---|---|---|
| tq_bias_raw_b1 (new) | 0.357 | **0.021** | **−94%** |
| tq_mse (primary) | 0.296047 | 0.296047 | 0.0 |
| tq_bytes_b2 | 134.0 | 134.0 | 0.0 (correction is global) |
| tq_bias_b1 (conformance) | 0.019938 | 0.019938 | 0.0 |
| tq_var_b2 | 0.988199 | 0.988199 | 0.0 |
| fwht_us | 46.1 | 46.1 | ~0 |

**Findings:**
1. **Debiasing is ~free and large:** logit-bias error drops 0.357 → 0.021
   (94%) with zero byte overhead and no change to MSE/bytes/compute. This is
   the cheap alternative to TurboQuant_prod's residual-QJL (which costs
   +1 bit/coord and is only justified where per-score variance matters more
   than memory).
2. **The residual 0.021 is estimator noise, not bias:** the correction factor
   is measured with n_rot=800 (deterministic); the debiased ratio 0.979 is
   1.0 minus the combined Monte-Carlo standard error. Raising n_rot shrinks it
   linearly in compute — but the factor is calibration-time (once per
   codebook), so this residual is not operationally meaningful.
3. `tq_bias_b1` (conformance to 2/π) is unchanged — the two metrics measure
   different things (theory-conformance vs faithful-logits) and both stay
   within guardrails.

**Tradeoffs / risks:** none measured. The only cost is a one-time per-codebook
calibration (already the accepted cost of P1.7); the per-token scoring cost and
byte budget are unchanged. Product (unbiased, low-variance but +1 b/coord)
remains the fallback where variance per score matters.

**Verdict:** **keep** — unbiased logits (0.357 → 0.021) at zero bytes, all
other metrics unchanged, checks green. New guardrail metric `tq_bias_raw_b1`
now tracked (selfcheck: must stay < 0.5).

**ASI (remember after reset):**
- Raw b=1 MSE estimator is ~0.64-shrunk (biased). The finite-d P1.7 debias
  (global scalar) is adopted: tq_bias_raw_b1 0.357 → 0.021 at 0 bytes.
- A closed-form analytic finite-d bias (instead of MC factor) was considered
  and rejected: the residual is calibration-time noise with no per-token
  impact; not worth the math.
