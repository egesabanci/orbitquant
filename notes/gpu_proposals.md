# GPU / Model-Dependent Proposals — Notes for Later Execution

These OrbitQuant proposals **cannot** be benchmarked with the pure-NumPy
statistical harness (no model, no GPU). They are deferred to a different box
with a decoder-only LLM and GPU. This file records what each needs so we can
catch up quickly.

> Status: **not implemented in the current effort.** These are notes only.

---

## P0.4 — Minimal KV-Cache Evaluation Harness
**Needs:** a small decoder-only model (e.g. Llama-3.1-8B-Instruct or smaller) to
capture real K/V tensors during prefill/decode.
**Must report:** key reconstruction error, value reconstruction error,
attention-logit bias/error, softmax KL vs full precision, attention-output
error, hidden-state error, generated-token quality on a tiny deterministic
task set, memory ratio including side information.
**Notes:** this is the bridge from synthetic stats to real attention. The
statistical baseline (P0.1-P0.3) validates the assumptions; P0.4 checks whether
they hold on real KV tensors.

## P2.1 — Fused Quantized Attention CUDA/Triton Kernel
**Needs:** GPU + CUDA/Triton toolchain.
**Goal:** do not materialize dequantized K/V. Compute `softmax(QK^T)V` in a
fused path reading packed quantized codes, reconstructing centroids in
registers, streaming through attention.
**Prerequisite:** P0.2/P0.5 must settle the representation (dense rotation vs
FWHT, octahedral decode, residual QJL, predictive reconstruction, pre-rotated
cache scoring) because the kernel depends on it.
**Notes:** this is where memory compression becomes serving speed.

## P3.1 — Full Token-Prediction Objective
**Needs:** a real model to compute next-token distributions.
**Goal:** move beyond softmax KL as a metric; directly optimize an approximate
downstream objective (KL between full-precision and quantized next-token
distributions, or a first-order surrogate over attention/MLP output changes).
**Notes:** may change which quantizer looks best, especially for values,
protected-token policies, and score-correction methods.

## P3.3 — Query Batching Optimization for Fused Kernels
**Needs:** fused kernel from P2.1.
**Goal:** load quantized codes once and compute attention for several queries in
one pass (compounds the memory-bandwidth win).
**Notes:** kernel scheduling optimization, not a quantizer. After P2.1.

## P3.4 — Lookup-Table Quantized Attention (b <= 4)
**Needs:** fused kernel + fast memory.
**Goal:** precompute per-query contribution tables per coordinate and centroid
index; attention score path becomes table lookup + accumulation instead of
multiply-heavy dequantization.
**Notes:** subcase of P2.1; only useful if the table fits in fast memory and
reduces real kernel time.

## P3.6 — RoPE Placement and Grouped-Head Rotation Experiments
**Needs:** a real model with RoPE (rotary positional embeddings) to test
rotation placement.
**Goal:** rotate before RoPE where possible, share/group rotations across
heads, measure whether RoPE reintroduces outliers after the transform.
**Notes:** grounded in RotateKV's pre-RoPE grouped-head rotation. Narrower
experiments than the full RoPE-commuting rotation problem (deferred).

---

## Out-of-scope summary (this effort)
- P0.4, P2.1, P3.1, P3.3, P3.4, P3.6 — need model/GPU, deferred.
- Any full-model or GPU evaluation.

## Statistically testable (implemented in this effort)
P0.1, P0.2, P0.3, P1.1-P1.8, P2.2-P2.10, P3.5, P3.7 — pure NumPy, no model/GPU.
