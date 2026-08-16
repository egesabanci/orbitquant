# 11 — Attention-Objective Probe: the Debias was a Surrogate Artifact

**Goal (self-originated, creative):** question the session's own assumption.
All prior estimator metrics (`tq_bias_raw`, `tq_bias_b1`) measure *pairwise*
inner-product bias — a proxy. The real thing TurboQuant exists for is softmax
attention over a context. Does the "better" we banked (the finite-`d`
multiplicative debias, exps 06/08) actually help attention, or did the proxy
mislead us? This is the strongest possible anti-overfit check: validate the
adopted design against the true objective.

**Method / Protocol (`baseline/benchmark_tq.attention_metrics`):** realistic
key set (uniform direction × lognormal norm) and query set, encoded under ONE
shared serving rotation P (the actual KV-cache situation — not the per-pair
Monte Carlo over many P used everywhere before). Every key stores the 6-bit log
norm + exact-Beta b-bit Lloyd-Max codes; a query scores the cache in the
rotated domain. Report softmax-KL (true vs estimated attention), recall@k, and
tail logit error (p95 |Δlogit| / mean-|true-logit|). Compare raw vs debiased
(multiplicative factor, the exp-06/08 default) vs debiased with a
*realistically-recalibrated* factor, across 3 seeds.

**Anti-cheat check:** nothing tuned to make the debias lose — all variants go
through the same probe; the realistic-calibrated factor is included to rule
out "wrong factor" as the cause; deterministic given seed; the probe models
the actual serving regime.

**Results — softmax-KL (mean over seeds):**

```
config      raw       debias(MC)   debias(realistic-c)
d=64  b=1  0.00437   0.00642    0.00679
d=64  b=2  0.00140   0.00154    0.00157
d=128 b=2  0.00074   0.00081    0.00084
```

New persistent metric `tq_attn_kl_b2` (b=2/d=64, adopted raw default):
**0.001245** (vs 0.001385 with the debias).

**Findings — why the debias hurts attention:**
1. **Softmax attention is noise-limited, not scale-limited.** Multiplying all
   estimated scores by 1/c to fix the global shrinkage amplifies the *per-key*
   logit-noise variance by 1/c² ≈ ×2.4 at b=1, ×1.25 at b=2. That noise is what
   softmax is sensitive to; the scale/temperature error it "fixes" is second
   order (raw c≈0.885 at b=2 is already near temperature-correct; at b=1 the
   temperature error is real but the ×2.4 noise blow-up dominates).
2. **Even a perfectly-calibrated factor loses.** Realistic-scale calibration
   (c=0.885 vs the adversarial-protocol's 0.895) does not rescue it — the loss
   is structural, confirming the mechanism, not a miscalibration.
3. **The pairwise-bias surrogate (2/π story) does not transfer to attention.**
   A global multiplicative shrinkage is (a) nearly invisible to softmax's
   ordering at moderate rates and (b) any correction transfers its cost to
   variance. Exps 06/08 optimized a surrogate; exp 11 shows the real objective
   prefers the untouched estimator.

**Action taken (honest, evidence-driven):** reverted `DEFAULT_DEBIAS` to
False. Surrogates correctly moved back up (tq_bias_raw 0.050→0.209, b1
0.028→0.357) — accepted as the rejected-proxy cost; the real objective
(tq_attn_kl_b2) improved; primary tq_mse and bytes unchanged; checks green.
The metric infra from 06/08 (tq_bias_raw) is retained as an honest diagnostic,
explicitly documented as non-actionable on its own.

**Tradeoffs / risks:** the probe is model-free (realistic random vectors), not
real KV tensors — the GPU box should re-validate on real attention. But the
synthetic serving setup is the right first word, and its mechanistic argument
(noise amplification by multiplicative correction) is general.

**Verdict:** **keep** — refutes an earlier kept default at the real objective;
adds the permanent guardrail `tq_attn_kl_b2`; corrects the session's
trajectory away from a surrogate; everything else unchanged and green.

**ASI (remember after reset):**
- NEVER optimize `tq_bias_raw` on its own; it rewarded a harmful change.
- The real objective for this domain is softmax attention: use
  `attention_metrics()` / `tq_attn_kl_b2`. Per-key logit-noise VARIANCE is the
  lever, not global scale.
- Multiplicative debias (and any global score scaling) amplifies noise by 1/c²;
  avoid it unless the estimator's variance is already negligible.
- Next creative avenue suggested by the noise-limited lens: per-key variance
  reduction (residual/QJL or more bits) and measure its attention-KL-per-byte
  tradeoff vs the raw b=2 design.
