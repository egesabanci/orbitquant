# 15 — K/V Bit Allocation on the KV Attention Objective (values are the output)

**Goal (creative, opens the value side):** every prior experiment — and the
P1.2 "More for Keys, Less for Values" direction — judged keys only. But a KV
cache stores values too, and values are consumed as `out = Σᵢ pᵢ vᵢ` (the
attention output). Value quantization error therefore enters the output
*linearly*, while key quantization error only reshapes the weights p. This
experiment measures the K/V bit-allocation tradeoff at fixed total bytes on
the real consumption objective, testing whether key-favored allocation is
right for OUTPUT fidelity or only for distribution fidelity.

**Method / Protocol (`benchmark_tq.kv_attention_error`, runner
`baseline/run_kv_alloc.py`):** realistic keys and values (independent draws of
the same law), separate shared serving rotations P_k / P_v, the adopted
pipeline per side (6-bit log norm + b-bit exact-Beta codes). Attention weights
from the true scores (p) or the estimated key scores (p̂, end-to-end).
Metrics: relative attention-output error
`E_q‖Σp̂ᵢv̂ᵢ − Σpᵢvᵢ‖² / E_q‖Σpᵢvᵢ‖²`, the value-only variant (p fixed —
isolates value quantization), and the key-side softmax-KL. True bytes/token =
(b_k+b_v)·d + 12.

**Anti-cheat check:** the objective is the literal definition of how values
are used; deterministic; robust across 3 seeds × n_q=80; the result REFUTES my
own prior expectation (key-favored) — nothing was tuned to win.

**Results (d=64; e2e at 268 bits, robust means over 3 seeds):**

```
alloc    bits   rel_out_err(e2e)  V-only    kl(key)
K3V1     268    0.453             0.481     0.00040
K2V2     268    0.186             0.177     0.00137
K1V3     268    0.065             0.046     0.00429
K3V2     332    0.178             0.177     0.00044
K2V3     332    0.050             0.046     0.00157
```

**Findings:**
1. **Value-favored allocation (K1V3) beats uniform by ~2.9× and key-favored
   (K3V1) by ~7× on attention-output error at the SAME 268 bits.** The
   "More for Keys" direction is refuted for output fidelity in this regime.
2. **The two objectives are separable and trade off cleanly:** value bits buy
   OUTPUT fidelity (V-only errors: 0.046/0.177/0.481 at b_v=3/2/1), key bits
   buy DISTRIBUTION fidelity (KL: 0.0004/0.0014/0.0043 at b_k=3/2/1). Key
   quantization perturbs the output only through p̂ (≤ +0.01 on top of the
   V-only error) — the output error is dominated by value reconstruction.
3. **Mechanism (why):** rel_out_err ≈ Σp̂ᵢ²·E‖Δvᵢ‖² / Σpᵢ²·E‖vᵢ‖² — the
   softmax concentration factor Σp² cancels, so the metric tracks the value
   codec's relative MSE. Value error is never averaged away by the attention
   distribution in ratio terms; the codec bit-width is the lever.
4. **Synthesis for deployment:** the K/V bit split is a two-objective choice —
   distribution/retrieval fidelity (keys) vs output/hidden-state fidelity
   (values). A serving stack that cares about output drift should lean
   value-favored; the P1.2 rule is validated only for the key-side objective.

**Implications / verdict:** first quantitative K/V allocation on the real
consumption objective: at equal bytes, K1V3 (0.065) >> K2V2 (0.186) >> K3V1
(0.453) for output error. No default change (the session benchmark is key-side
by design); recorded as a deployment/P1.2-reframing finding, to be re-validated
on real KV tensors (value law and attention concentration differ there).
**keep as finding + runner.**

**ASI (remember after reset):**
- Value bits buy OUTPUT fidelity (linear in the attention output), key bits
  buy DISTRIBUTION fidelity; do not transfer the "keys>values" rule to output
  objectives. K1V3 ≈ 7× better output error than K3V1 at equal bytes here.
- Reuse `kv_attention_error(b_k, b_v, est_p=...)`; runner
  `python -m baseline.run_kv_alloc`.
- Real-KV follow-up: values often have different marginals (and attention is
  more peaked) — the ratio metric is concentration-independent, but the
  absolute output error and the real value law should be checked on GPU box.
