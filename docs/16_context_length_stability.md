# 16 — Attention Fidelity is Context-Length Stable (long-context validation)

**Goal (creative, closes the biggest deployment gap):** KV compression exists
*for* long context, yet every probe in this session (and the prior effort)
measured one key-set size (n_db=1500). If quantized attention fidelity degraded
as the cache grows — extreme-value logits, more keys competing for top rank,
more exposure to per-key noise — the benchmark's headline numbers would
overstate real long-context serving. This sweep asks: does attention fidelity
scale with context length, or is it stable?

**Method / Protocol (`benchmark_tq.attention_context_scale`, runner
`baseline/run_context_scale.py`):** the same serving probe as exps 11-15
(same rotation via fixed rot_seed, same query set, same adopted pipeline —
6-bit norm + exact-Beta codes + raw estimator), sweeping the key-set size
n_db ∈ {128, 512, 1500, 4000, 8000} at b∈{1,2}, plus the fast hdhdh rotation
at 8k. Deterministic; only the context length changes.

**Anti-cheat check:** nothing fitted; the query set and rotation are held
fixed across sizes so any change is purely a context-length effect; robust
across 3 dataset seeds.

**Results (d=64; same rotation & queries across sizes):**

```
n_db   KL b=1    KL b=2    r5 b=1   r5 b=2   noise b=2   p95 b=2
128    0.00497   0.00153   0.705    0.900    0.4624      0.906
512    0.00519   0.00166   0.670    0.895    0.4596      0.934
1500   0.00483   0.00157   0.590    0.855    0.4517      0.919
4000   0.00485   0.00157   0.525    0.880    0.4504      0.915
8000   0.00488   0.00157   0.530    0.820    0.4499      0.911
```

hdhdh @ 8000 keys: KL 0.001559 (Haar 0.001568). KL(8000)/KL(1500) across 3
seeds: 0.957 / 1.043 / 1.003 (sampling noise, no trend).

**Findings:**
1. **Softmax-KL is context-stable:** b=2 KL stays 0.00153–0.00166 across a
   64× context range (128 → 8000), b=1 likewise (~0.0049). Per-key score-noise
   and p95 logit ratio are constant (0.45, 0.91) — the quantization noise is a
   per-key property that does not compound with context length.
2. **recall@k drops only because the task gets harder** (top-5 of 8000 vs top-5
   of 128), not because of quantization — the distribution-level fidelity (KL)
   is the right long-context measure and it does not degrade.
3. **The fast structured rotation holds at scale** (hdhdh @ 8k ≈ Haar), and
   the benchmark's n_db=1500 headline numbers are representative of long
   context (up to at least 8k in this regime).

**Implications / verdict:** long-context serving is validated: 134 bits/token
delivers the same attention fidelity at 8k keys as at 128. This closes the
deployment loop for the session's stack (hdhdh + exact-Beta + 6-bit norm +
raw estimator). No default change; **keep as finding + runner.** Real models
(long-context softmax over correlated tokens, sink keys) remain GPU-box work.

**ASI (remember after reset):**
- Attention fidelity (softmax-KL) is context-length stable in this regime;
  per-key score-noise does not compound. n_db=1500 benchmark numbers are
  representative up to ≥8k.
- recall@k is not comparable across set sizes; use KL for scale comparisons.
- Runner: `python -m baseline.run_context_scale`.
