# 17 — Attention-Fidelity Tail Risk: Query Norm Drives the Tail

**Goal (creative, risk-focused):** every attention metric in this session is a
MEAN. But serving quality is decided by tails — a few badly-served queries can
dominate real harm in long-context generation. This experiment maps the
*distribution* of per-query attention fidelity and asks: is quantized attention
risk concentrated, what drives it, and does data-oblivious protection (exp 13)
help the tail more than the mean?

**Method / Protocol (`benchmark_tq.attention_kl_tail`):** per-query softmax-KL
over n_q=120 realistic queries (n_db=1500 keys, shared Haar serving rotation,
adopted pipeline) at b∈{1,2,3}, reporting mean/p50/p90/p95/p99 and the
correlation of per-query KL with query norm. The same distribution for the
top-1%-by-stored-norm protection variant (exp 13) isolates the tail benefit.

**Anti-cheat check:** same seeded probe; no fitting; the tail statistics are
standard robust quantities; the query-norm correlation is measured, not
assumed.

**Results (d=64, n_q=120):**

```
b   mean      p50       p90       p95       p99       qnorm-corr
1   0.005382  0.003649  0.008796  0.012178  0.041080  +0.942
2   0.001743  0.001171  0.002885  0.004156  0.013204  +0.951
3   0.000510  0.000336  0.000855  0.001196  0.003658  +0.930
```

Protection (top-1% keys b=3, rest b=2, same avg bytes): mean −4.0%, p95
−7.7%, p99 −4.2%.

**Findings:**
1. **Attention fidelity has a heavy per-query tail:** at b=2 the p99 KL is
   10.6× the median (0.0132 vs 0.0012); the mean is inflated ~1.5× over the
   median. The benchmark's mean guardrail hides this: a small fraction of
   queries are served an order of magnitude worse.
2. **The tail is driven by QUERY NORM (corr +0.94–0.95):** high-norm queries
   produce sharp attention (logits scaled up → low effective temperature), and
   the same per-key score noise then moves much more probability mass. The
   mechanism is generic (query scale × key noise), not an artifact of the
   input law.
3. **Protection is partially tail-limited:** it cuts the middle tail more than
   the mean (p95 −7.7% vs mean −4.0%) but barely touches the extreme tail
   (p99 −4.2%) — extreme sharpness amplifies even the protected keys' residual
   noise. So exp-13's pool is a modest risk mitigant, not a tail cure.
4. **Deployment implication:** serving stacks should (a) expect worse quantized
   attention on high-norm queries (a real-model check of query-norm
   distributions is a GPU-box item), and (b) track a tail metric, not just the
   mean — hence the new guardrail `tq_attn_kl_b2_p95 = 0.004156`.

**Implications / verdict:** the risk structure of quantized attention is now
known: heavy-tailed in the query dimension, query-norm-driven, only partially
mitigable by key-side protection. New permanent guardrail metric
`tq_attn_kl_b2_p95` (p95 of per-query KL, b=2/d=64). No default change;
**keep as finding + guardrail.**

**ASI (remember after reset):**
- Per-query attention KL is heavy-tailed (p99 ≈ 7.6× mean) and ~+0.94
  correlated with query norm (sharp attention amplifies per-key noise).
- Mean metrics understate risk; track tq_attn_kl_b2_p95 (0.004156 now).
- Protection (exp 13) helps the middle tail (p95 −7.7%) more than the mean,
  but not the extreme tail (p99 −4.2%).
- Runner: `attention_kl_tail(b=2, n_q=120, frac=0.01, ...)`.
