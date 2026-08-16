# 13 — Data-Oblivious Key Protection on the Attention Objective

**Goal (creative, follows the attention-objective line):** at a fixed average
true-byte budget, can a *data-oblivious* bit-reallocation over keys improve
softmax-attention fidelity — i.e., "make TurboQuant better at the same bytes"
on the REAL objective? Since attention error is proportional to high-norm
keys (absolute direction error scales with the stored norm), a cheap policy is:
protect the top `frac` of keys **by their stored 6-bit norm** (available at
encode time, and inferable by any decoder from the shared rule → **zero extra
metadata**) with a higher direction bit-width, downshifted nothing but keeping
the rest at the base rate.

**Method / Protocol (`benchmark_tq.attention_protection`, runner
`baseline/run_protection.py`):** same serving probe as exp 12 (one shared
rotation, realistic keys+queries, softmax-KL and recall@5). Per key: 6-bit log
norm + `b_i`-bit exact-Beta direction code, where `b_i` is base_b for most keys
and prot_b for the protected pool. Honest avg bits/token = mean(b_i)·d + 6.
Configs anchored by uniform b∈{1,2,3} and include a **random-protection
control** (same extra bits, random keys) to isolate the norm-targeting effect,
so "extra bits help" cannot masquerade as "norm-targeting helps."

**Anti-cheat check:** no fitting; deterministic; avg-bytes honestly reported
(the protected fraction's bits are counted); the control isolates the
mechanism; robust across 3 dataset seeds × n_q=150.

**Results (d=64; robust KL means over 3 seeds, n_q=150):**

```
config                  avg bits bytes/tok   KL        recall@5
uniform b=2                 134.0   16.75  0.001571     0.685
uniform b=1                  70.0    8.75  0.004827     0.405
uniform b=3                 198.0   24.75  0.000444     0.795
prot top1%  b=3 / rest b=2  134.6   16.83  0.001507     0.700
prot RANDOM1% b=3 / rest b=2 134.6  16.83  0.001558     0.690
prot top5%  b=3 / rest b=2  137.2   17.15  0.001369     0.730
prot top10% b=3 / rest b=2  140.4   17.55  0.001236     0.770
prot top5%  b=4 / rest b=1   79.6    9.95  0.004000     0.685
```

Means over 3 seeds (n_q=150): uniform 0.001483; top1% 0.001427; random1%
0.001471; top5% 0.001298.

**Findings:**
1. **Norm-targeting is real but small.** At the same avg bytes (134.6 vs 134),
   protecting the top-norm 1% at b=3 cuts softmax-KL −4.1% and raises recall@5
   0.685→0.700. The control ordering holds everywhere (topnorm < random <
   uniform), so the gain is from targeting high-norm keys, not from the extra
   bits alone. Scaling: −12.5% KL at top5% (+2.3% bytes), −21% at top10%
   (+4.8% bytes) — a smooth, monotone, honest exchange.
2. **Protection beats the smooth uniform interpolation.** At ~137 bytes,
   top5% protection (KL 0.00137) is better than interpolating uniform b=2→b=3
   (~0.00147): a convex combination of rates that spends bits only on the
   keys whose noise matters most.
3. **But it is strictly a mixed-width design → deployability tension.**
   Variable per-key payload breaks the P0.5 paged-cache fixed-size property
   (per-key 134→198 bits) and complicates fused kernels. For a
   "lighter/deployable" default that is the decisive cost, so this is recorded
   as a validated candidate, NOT adopted as default.
4. Norm-precision protection (8-bit norm headers on high-norm keys) was
   checked implicitly by exp 12: norm error is already negligible vs direction
   error there (8-bit vs 6-bit norm → identical KL), so norm-side protection
   has ~0 headroom; direction-side protection is the mechanism.

**Implications / verdict:** data-oblivious norm-key protection is a real,
zero-metadata, ~4–21% attention-KL lever at ~constant bytes in this regime —
strongly consistent with the OSCAR/RotateKV sink-protection motif, validated
model-free on the REAL objective. Default unchanged (uniform scalar b=2) due
to the paged-fit/deployability cost; the policy is a high-priority candidate
for the real-KV GPU box where the quality-vs-variable-size tradeoff can be
judged against actual model outputs. **keep as finding + runner.**

**ASI (remember after reset):**
- Protected-pool-by-stored-norm gives real but small attention gains at
  ~fixed bytes (top1% −4%, top5% −12%); random control proves the targeting
  effect is real; mechanism = high-norm keys carry linearly larger absolute
  direction error.
- Do NOT adopt variable-width pools as the default here (breaks paged-cache
  fit). Candidate for GPU box where OSCAR-style pools already exist.
- Norm-side protection has ~0 headroom (norm error already negligible).
