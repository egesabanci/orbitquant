# 02 — Vectorized FWHT Butterfly (compute lightness)

**Goal:** make the structured (hdhdh) rotation — the P0.2 fast alternative to
dense Haar — actually fast in pure NumPy. The P0.1 report showed the pure-Python
per-block FWHT butterfly was ~65× slower than BLAS dense matmul; this is the
established, provably-equivalent open target for the "lighter" axis
(`fwht_us`).

**Method / Protocol:** `baseline/rotations.py::fwht` is the O(d log d) Walsh-
Hadamard butterfly. Replaced the inner per-block Python loop
(`for i in range(0, d, h*2): ...`) with a **vectorized** level: each level is a
strided `reshape(-1, 2h)` to view the 2h-blocks, slice out the two halves, and
write `a+b` / `a-b` in place. The per-element arithmetic is the very same
add/subtract on the same float values, so the output is **bitwise identical** to
the scalar reference — a strict algebraic rewrite (documented in the docstring),
not a different transform. No other code changes; `_hadamard_apply`,
`hadamard_sign_flip`, rotations, and the benchmark are untouched.

**Anti-cheat check:** no seeds, inputs, budgets, or metric definitions changed.
Equivalence proven by `np.array_equal` against the scalar loop over 20 random
inputs for d ∈ {4, 8, 16, 64, 128, 512}, plus the analytic H@x check, plus the
benchmark's own invariants (`fwht(fwht(x)) == d·x`, hdhdh orthogonality) — all
green. Quality metrics are seeded-deterministic and unchanged by construction.

**Results (before → after):**

| Metric | Before | After | Δ |
|---|---|---|---|
| tq_mse (primary) | 0.295923 | 0.295923 | 0.0 (identical) |
| tq_bytes_b2 | 136.0 | 136.0 | 0.0 |
| tq_bias_b1 | 0.019938 | 0.019938 | 0.0 |
| tq_var_b2 | 0.988199 | 0.988199 | 0.0 |
| fwht_us | ~389 (372–398) | **46.1** | **−88% (8.4×)** |

Per-level timing (best-of, µs): `fwht` loop 124 → vec 14 (d=128); scaling is
near-linear as O(d log d) — d=512: 23µs, d=2048: 46µs.

**Findings:**
1. A one-level vectorized FWHT is 8.4× faster end-to-end (hdhdh 387→46µs) with
   provably identical output — zero risk, pure win. It repairs the P0.1 runtime
   caveat that made the structured rotation impractical in pure Python.
2. The 14µs floor at d=128 is mostly ~7 numpy call overheads (one per level);
   pushing further in numpy has diminishing returns. At d=128 BLAS dense is
   still ~7× faster (1.9µs); the FWHT wins at d ≥ ~512 and is the right choice
   on GPU and for O(d log d) scaling — the P0.2 claim is now honest at both the
   algorithmic and the NumPy-practical level.
3. A ping-pong buffered variant was tried and measured slightly slower
   (16 vs 14µs); not adopted — simpler code won.

**Tradeoffs / risks:** none for quality (bitwise identical). The only residual
caveat is that single-vector FWHT in NumPy at small head dims (64–256) is still
beaten by BLAS dense for *one* rotation; the structured transform's value is
algorithmic complexity, tiny parameter storage, and GPU friendliness, not
beating BLAS at d=64 per-call.

**Verdict:** **keep** — fwht_us 389 → 46 (−88%), every other metric bitwise
unchanged, all checks green.

**ASI (remember after reset):**
- The move that won: vectorize each butterfly level as `m = x.reshape(-1, 2h);
  a = m[:, :h].copy(); m[:, :h] = a+b; m[:, h:] = a-b`. Bitwise-equal to the
  loop. 8.4× on the measured metric.
- Next: norm side-info bytes are the cleanest remaining lightness win
  (tq_bytes_b2 = 136, all 8 bits are the log-norm header).
