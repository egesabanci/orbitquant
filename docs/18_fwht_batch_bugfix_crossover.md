# 18 — FWHT Batch-Safety Bugfix + the Dense-vs-Structured Crossover (NumPy)

**Goal (creative, engineering + honesty):** complete the compute-lightness
story with the OTHER serving side — the key-cache ENCODE path (n keys at
once). Hypothesis: the vectorized FWHT butterfly is batch-safe and should be
measured for whole-cache encoding. Testing it exposed a real latent bug and a
quantitative verdict that tempers the exp-02/14 rotation story with evidence.

**Method / Protocol:** applied `rotations.fwht` / `hadamard_sign_flip` to
(n, d) arrays and compared bitwise against per-row application; measured
per-key cost for batched structured rotation vs dense matmul at n ∈ {2048,
8192}, d=128.

**Anti-cheat check:** bitwise identity is the equivalence contract (strict
algebraic rewrite); timing is best-of; the verdict (dense wins) was not the
one I expected — reported as measured.

**Findings:**
1. **Real bug (fixed): batch application was silently wrong.** `fwht` used
   `d = x.shape[0]` (the BATCH size for (n,d) inputs) as its butterfly loop
   bound, and `_hadamard_apply` normalized by `sqrt(x.shape[0])` instead of
   `sqrt(d)` — a (n, d) input produced garbage (roundtrip failed at n=3). No
   existing caller hit it (all paths loop per-row or materialize the matrix),
   but the Rotation API advertised vector transforms and would have silently
   corrupted whole-cache use. Fixed to `shape[-1]` (identical for 1-D);
   `random_permutation` also made batch-safe (`x[..., perm]`). Verified
   bitwise: batch == per-row for n∈{3, 64, 1024}, roundtrip exact. Guardrail
   added to `benchmark_tq --selfcheck` so batch safety cannot regress.
2. **Quantitative crossover: dense BLAS dominates in pure NumPy at every
   practical size.** Single vector: dense 1.9µs vs hdhdh 46µs (24×). Cache
   encode (n=8192, d=128): dense matmul 0.088µs/key vs batched hdhdh 5.0µs/key
   (~55×). The FWHT's O(d log d) advantage is swamped by NumPy per-op overhead
   and memory traffic — it is a GPU/vectorized-hardware property, not
   observable in pure Python (consistent with, and now quantified beyond, the
   P0.1 caveat).
3. **Honest recommendation:** in the NumPy serving simulation, the fast
   rotation is the parameter-light GPU-kernel choice (P2.1 target), not the
   NumPy fast path — for NumPy-scale experiments, dense matmul (or the
   materialized hdhdh matrix, exp 14) is the practical rotation. The
   exp-02 fwht_us metric (its own cost, 8.4× improved) remains valid; it never
   claimed to beat BLAS.

**Implications / verdict:** fixed a real batch-safety bug (kept, guarded);
documented that pure-NumPy compute favors dense rotation at all practical
sizes, with the structured rotation's value reserved for GPU kernels. No
metric change (fwht_us unchanged 45.9µs); **keep as bugfix + finding.**

**ASI (remember after reset):**
- fwht/_hadamard_apply/perm are now batch-safe (shape[-1] / x[..., perm]);
  selfcheck guards batch==per-row. Do NOT reintroduce shape[0] on the last
  axis.
- In NumPy, dense matmul is the fast rotation (0.088µs/key vs 5µs/key at
  d=128, ~55×); the FWHT win is GPU-only. Documented crossover.
