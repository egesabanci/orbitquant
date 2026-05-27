# TurboQuant Optimization Proposals

> **Updated 2026-05-27** — Re-audited against TurboQuant, QJL, PolarQuant, and recent KV-cache quantization work. The list now separates foundational measurement infrastructure from algorithm ideas, corrects places where Haar-rotation theory was being over-applied to structured transforms, adds stronger outlier/rotation/budgeting proposals, and moves weaker active ideas to Deferred / Not Recommended.

---

## P0 — Foundational: Implement First

These items are needed before the rest of the backlog can be evaluated honestly. The immediate risk is not lack of ideas; it is accepting a structured-transform or low-bit trick before measuring whether it preserves the assumptions TurboQuant relies on.

### P0.1 — Reference Implementation and Rotation Validation Harness

Build a pure PyTorch/CPU reference implementation of `TurboQuant_mse`, `TurboQuant_prod`, and QJL, then add a validation harness for rotations and codebooks.

The harness should compare dense Haar random rotation, dense Gaussian-QR rotation, Hadamard/sign-flip variants, and no rotation on:

- coordinate distribution fit against the finite-d Beta law;
- coordinate covariance and higher-order dependence;
- MSE distortion for `b = 1..4`;
- inner-product bias and variance;
- attention-logit error on real KV tensors;
- runtime and memory overhead.

This should be the first experiment because several later proposals depend on structured rotations behaving "random enough" in practical head dimensions.

> **Paper grounding:** TurboQuant's guarantees come from random rotation producing a uniformly random point on the sphere, whose coordinates follow Lemma 1's Beta distribution. Structured transforms may work very well in practice, but they do not inherit that exact guarantee automatically.

### P0.2 — Structured Fast Rotations (Hadamard + Diagonal Sign Flip)

Replace the dense `O(d^2)` random rotation with a fast structured transform such as `D1 H D2 H D3 H`, where `H` is a normalized Walsh-Hadamard transform and each `D` is a random diagonal sign matrix. This gives `O(d log d)` compute, tiny parameter storage, and a GPU-friendly butterfly structure.

The claim should be empirical, not overstated: a Hadamard/sign-flip stack is orthogonal and mixes coordinates, but it is not the same distribution as a Haar random rotation. Test single-round HD, multi-round HDHD/HDHDH, optional random permutation, and the outlier-aware permutation in P1.4.

> **Paper grounding:** TurboQuant uses a random rotation to make coordinates Beta-distributed and nearly independent. PolarQuant and recent rotation-based KV-cache methods show fast Hadamard-style preconditioning is practical, but the exact distortion bounds need validation under the structured replacement.

### P0.3 — Exact Finite-Dimension Codebooks

Precompute Lloyd-Max codebooks using the exact finite-dimensional coordinate distribution from TurboQuant Lemma 1:

```text
f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
```

for the actual head dimensions used by target models, especially `d = 64` and `d = 128`.

For Haar rotations, these are the theoretically correct scalar codebooks. For structured rotations, compare them against empirical codebooks learned from rotated calibration tensors. If the empirical codebook wins materially, keep both paths: exact-Beta for theory and empirical for production experiments.

> **Paper grounding:** TurboQuant Theorem 1 gives `D_mse = d * C(f_X, b)`. The scalar quantizer quality depends directly on matching `f_X`; the Gaussian approximation is weakest at low bit-widths and small practical head dimensions.

### P0.4 — Minimal KV-Cache Evaluation Harness

Add a small decoder-only model integration that can capture real K/V tensors, quantize them during prefill/decode, and report:

- key reconstruction error;
- value reconstruction error;
- attention-logit bias/error;
- attention output error;
- generated-token quality on a tiny deterministic task set;
- memory ratio including side information.

Synthetic vector tests are necessary but insufficient. Several proposals optimize MSE while the real failure mode may be attention-logit drift or value aggregation error.

---

## P1 — High Impact, Low Implementation Risk

These should come after P0 because they use the same reference implementation and evaluation harness. They are mostly policy and calibration changes, not kernel work.

### P1.1 — Logit-Error Mixed Mode Across Layers and Heads

Use `TurboQuant_mse` where it is empirically safe and `TurboQuant_prod` where unbiased inner-product estimation is actually needed. The decision rule should be based on measured attention-logit bias/error per layer and head, not only on outlier presence.

The earlier version of this idea overclaimed that shallow layers without outliers do not need unbiased correction. That is too strong: `TurboQuant_mse` can be biased even without visible outliers, especially at low bit-width. The better test is direct logit error under cached queries.

> **Paper grounding:** TurboQuant Section 3.2 proves MSE-optimal quantizers are biased for inner products. QJL Figure 2 shows outliers are layer-dependent. Combining both suggests a measured mixed-mode policy rather than a fixed all-layer choice.

### P1.2 — Separate Key/Value Objectives with Key-Favored Bit Budgets

Keys and values need different objectives. Keys control attention logits, so key quantization should prioritize low inner-product error and low bias. Values are averaged by attention weights, so they need good reconstruction under the actual attention distribution.

Start with `TurboQuant_prod` or debiased MSE for keys and `TurboQuant_mse` or attention-weighted MSE for values. Also test asymmetric bit budgets such as `K=4, V=2` against the reverse allocation at the same average KV size.

> **External grounding:** [KV-AdaQuant / "More for Keys, Less for Values"](https://arxiv.org/abs/2502.15075) reports that keys are more sensitive to quantization than values and that assigning more bits to keys can strongly outperform the reverse allocation at equal budget.

### P1.3 — Quantized Norm and Radius Storage

The papers remove coordinate scale/zero-point overhead, but still store scalar side information: QJL stores key norms, PolarQuant stores radii, and TurboQuant_prod stores residual norms.

Quantize these scalars in log space with a small shared format, for example 8-bit log-scale with per-layer/head calibration. Measure both quality and true memory ratio. This is low-risk because it does not change the vector code itself.

> **Paper grounding:** QJL stores `(sign(Sk_i), ||k_i||_2)`. TurboQuant_prod stores a residual norm. PolarQuant stores radius. These are small per token, but persistent at long context.

### P1.4 — Outlier-Aware Hadamard Channel Permutation

Before applying a Hadamard transform, reorder channels so persistent outlier channels are spread across different butterfly groups. This keeps the speed of FWHT while reducing the chance that a few large channels dominate a local mixing path.

Use a calibration pass to rank channels by magnitude or contribution to logit error, then generate a fixed per-layer/head permutation. Compare random permutation, bit-reversal permutation, and outlier-aware permutation.

> **External grounding:** [RotateKV](https://arxiv.org/abs/2501.16383) uses channel reordering with FWHT-style rotations to adapt to channel-wise outlier distributions while preserving fast structured rotation.

### P1.5 — Attention-Sink, Recent-Token, and Outlier-Token Protection

Keep a tiny subset of tokens at higher precision: attention sinks, system/prefix tokens, delimiter-like tokens, very recent tokens, and dynamically detected outlier tokens. The protected pool should be tiny enough that memory ratio barely moves.

This is a narrower and more useful version of age/role-aware precision. It should be implemented as a cache-write policy with a fixed high-precision quota, then tested against uniform quantization at the same memory budget.

> **External grounding:** [RotateKV](https://arxiv.org/abs/2501.16383) uses attention-sink-aware quantization, and [outlier-token tracing work](https://arxiv.org/abs/2505.10938) reports that a small subset of unusual tokens can dominate KV quantization error.

### P1.6 — Adaptive Bit Allocation Across Layers and Heads

Once P1.1 has per-layer/per-head error measurements, allocate bits where they buy the most quality: deeper or fragile layers, sensitive heads, and outlier-heavy heads get more bits; easy layers/heads get fewer.

The total bit budget stays fixed. The experiment is a rate-distortion scheduler over existing quantizers rather than a new quantizer.

> **Relationship to P1.1:** P1.1 chooses MSE vs product mode. P1.6 generalizes that into a mixed-precision policy.

---

## P2 — Strong Candidates, Moderate Effort

These are worth testing after the reference path is stable. They either require additional math checks, calibration data, or systems work.

### P2.1 — Fused Quantized Attention CUDA Kernel

Do not materialize dequantized K/V tensors. Compute `softmax(QK^T)V` in a fused path that reads packed quantized codes, reconstructs centroid values in registers, and streams through attention.

This is where memory compression becomes serving speed. The reference implementation should first prove quality; the fused kernel then proves deployability.

> **Prerequisite:** P0.2 should settle the rotation structure because the kernel needs to know whether it must apply dense rotation, FWHT, or pre-rotated-cache scoring.

### P2.2 — Fast Structured Residual QJL

TurboQuant_prod removes inner-product bias by applying QJL to the residual, but the residual sketch matrix is still dense in the paper algorithm. Test a structured residual sketch using SRHT/FWHT-style rows with random signs and optional row subsampling.

The first requirement is statistical: verify unbiasedness or quantify any bias under the structured replacement. The second is systems: determine whether residual sketching becomes cheap enough to use broadly.

> **Paper grounding:** QJL's estimator depends on random projection rows and sign bits. Structured JL transforms are plausible replacements, but they need explicit bias/variance tests.

### P2.3 — Orthogonalized Residual Sketches

QJL reports practical gains from orthogonalizing JL rows. Apply the same idea to TurboQuant's residual QJL stage and compare iid Gaussian rows, orthogonalized Gaussian rows, and structured orthogonal rows.

This has no extra stored bits. It may reduce estimator variance, but the scaling and unbiasedness should be verified because orthogonalized rows are not literally iid Gaussian samples.

> **Paper grounding:** QJL Section 4.1 reports that orthogonalized random Gaussian matrices consistently improve practical performance.

### P2.4 — Variable-Size Residual QJL

Make the residual sketch dimension `m` a tunable parameter instead of fixing `m = d`. Under a total budget

```text
B = b_mse * d + m
```

search for the best split between scalar MSE bits and residual sign bits.

For `m < d`, memory falls but residual estimator variance rises. For `m > d`, the residual stage may beat adding another scalar bit in fragile layers. The reference implementation should directly plot this tradeoff.

> **Paper grounding:** TurboQuant Theorem 2 bounds product distortion through residual MSE and QJL variance. QJL's estimator is an average over projection rows, so changing row count naturally changes variance.

### P2.5 — Covariance- and Attention-Aware Offline Rotations

Replace the vague "learned deterministic rotation" idea with a concrete calibration method: estimate key covariance, value covariance, and attention-weighted error sensitivity offline, then derive fixed rotations and clipping thresholds per layer/head.

This keeps serving-time cost predictable while allowing the transform to align with actual model statistics rather than worst-case spherical inputs.

> **External grounding:** [OSCAR](https://arxiv.org/abs/2605.17757) estimates attention-aware covariance structures offline and uses them to derive deployable rotations and clipping thresholds for low-bit KV-cache quantization.

### P2.6 — Importance-Weighted Coordinate Bit Allocation

After rotation, coordinates are close to independent in the TurboQuant story, but they may not be equally important for real attention. Estimate per-coordinate sensitivity from calibration queries and allocate more bits to coordinates that drive logit or output error.

This should be tested after P0.2/P1.4 because the optimal coordinate budget depends on the chosen rotation and channel permutation.

> **Paper grounding:** TurboQuant's scalar quantization is separable after rotation. If real rotated coordinates have unequal empirical variance or sensitivity, separable mixed precision is the natural extension.

### P2.7 — Clipped or Companded Lloyd-Max Codebooks

Structured rotations and real KV tensors may have heavier tails than the exact Beta model. Test low-bit codebooks with calibrated clipping or companding before Lloyd-Max quantization.

This is especially relevant for `b = 1` and `b = 2`, where a few tail coordinates can dominate error. The test should compare exact-Beta codebooks, empirical unclipped codebooks, clipped empirical codebooks, and log/µ-law style companding.

> **Relationship to P0.3:** P0.3 is the clean theoretical codebook. P2.7 is the practical fallback when the measured distribution does not match the theoretical one.

---

## P3 — Worth Testing Later

These are promising but should not block the first implementation cycle.

### P3.1 — Distortion Metric Aligned with Token Prediction

Move beyond raw MSE and inner-product error by approximating the downstream objective: KL divergence between full-precision and quantized next-token distributions.

A first-order softmax expansion can yield an attention-weighted inner-product metric. This may change which quantizer looks best, especially for values and protected-token policies.

### P3.2 — Analytical Debiasing for TurboQuant MSE

`TurboQuant_mse` is biased for inner-product estimation at low bit-width. Precompute codebook-specific scalar correction factors, or a small per-bucket correction table, to reduce bias during scoring without storing residual QJL bits.

This is attractive for latency-sensitive paths where `TurboQuant_prod` is too expensive. It should be judged by logit bias and downstream quality, not just synthetic inner-product bias.

### P3.3 — Query Batching Optimization for Fused Kernels

When multiple queries hit the same KV cache, load quantized codes once and compute attention for several queries in one pass. This compounds the memory-bandwidth win from KV compression.

This belongs after P2.1 because it is a kernel scheduling optimization, not a quantizer.

### P3.4 — Lookup-Table Quantized Attention for Low Bit-Widths

For `b <= 4`, precompute per-query contribution tables for each coordinate and centroid index. The attention score path becomes table lookup plus accumulation instead of multiply-heavy dequantization.

This should be treated as a subcase of P2.1. It is only useful if the table fits in fast memory and reduces real kernel time.

### P3.5 — Block-Structured Hybrid Rotation

If pure Hadamard/sign-flip rotations underperform dense Haar rotations, add a small dense mixing layer after the fast transform, for example 8x8 or 16x16 orthogonal blocks.

This preserves most of the `O(d log d)` benefit while adding local mixing capacity. It should only be tested after P0.2 proves where structured rotations fail.

### P3.6 — RoPE Placement and Grouped-Head Rotation Experiments

Full RoPE-commuting rotations are still a hard mathematical problem, but narrower placement experiments are concrete enough to test: rotate before RoPE where possible, share/group rotations across heads, and measure whether RoPE reintroduces outliers after the transform.

> **External grounding:** [RotateKV](https://arxiv.org/abs/2501.16383)'s pre-RoPE grouped-head rotation is a practical signal that this can be evaluated experimentally without solving the full commuting-rotation problem.

### P3.7 — ANN-Specific TurboQuant Scoring Path

For nearest-neighbor search, score directly over TurboQuant codes and rerank only a small candidate set in full precision or higher precision.

TurboQuant already reports strong recall and near-zero indexing time. The next systems win is avoiding full dequantization during search.

---

## Deferred / Not Recommended

These ideas are either invalid as currently stated, redundant with stronger proposals, premature, or likely to lose to simpler baselines.

### Joint Vector Quantization on Rotated Coordinates

The random rotation's purpose is to make coordinates nearly independent. If that works, joint quantization has little theoretical benefit and large codebook growth. Revisit only if P0.1 shows strong residual correlation after the chosen rotation.

### Multi-Stage Residual Quantization

The two-stage MSE + QJL design is already clean and near-optimal. Additional residual stages add complexity unless measured residual structure remains after QJL.

### DCT/DST Frequency-Domain Transform

Hadamard-style transforms are the stronger baseline: cheaper, more common in quantization systems, and better connected to JL-style mixing. DCT/DST should not be a separate active branch without evidence.

### Entropy Coding or RLE on Quantized Indices

TurboQuant itself estimates only modest entropy-coding savings, around 5% for one reported setting, and RLE is especially suspect because good rotations should make adjacent coordinates close to independent. Decode complexity and kernel friction likely outweigh the memory win. Revisit only after packed-code kernels exist.

### Dithered or Stochastic Quantization

Subtractive dithering is natural for uniform scalar quantizers, but TurboQuant uses Lloyd-Max centroid quantization. Adding dither introduces state and variance for unclear benefit. Keep it out of the active path until deterministic baselines are exhausted.

### Query-Conditional Adaptive Bit Width for Real-Time Generation

A coarse pass followed by selective refinement requires multiple scans of the KV cache per generated token. That is usually the wrong tradeoff for autoregressive serving. It may still be useful for offline ANN search.

### Residual QJL Coordinate Subsampling

As written, this duplicates P2.4. The clean formulation is variable-size QJL with `m` projection rows. Keep one proposal, not two.

### Online Adaptive Bit-Width via EWMA

Changing bit-widths during generation complicates cache layout, packed kernels, and old-token compatibility. Static or block-static calibration should be tested first.

### Runtime Codebook Refinement via Streaming k-Means

Multiple codebook versions inside one live KV cache create decode and metadata complexity, and online refinement weakens TurboQuant's data-oblivious advantage. Empirical codebooks from offline calibration are the cleaner test.

### Norm-Baked Scalar Quantizer

This does not really eliminate norm information unless the representation also stores or infers a norm bucket. In practice it turns side information into codebook-version metadata. P1.3 is simpler.

### Quantized Dense Rotation Matrix Storage

If structured rotations win, dense matrix storage disappears. If dense rotations remain necessary, compute cost is the larger problem. This is not a high-leverage experiment.

### Asymmetric Query/Key Transformations Beyond QJL

QJL already gives a concrete asymmetric estimator. Generalizing to unrelated query/key transforms needs a mathematical construction before it is experimentally meaningful.

### Cache-Line-Aware Code Layout

Premature before there is a fused quantized attention kernel and profiler data. Revisit after P2.1.

### Cross-Head or Cross-Layer Codebook Sharing

The codebooks are tiny compared with the KV index arrays. This optimizes the wrong memory term.

### Quantization-Aware Prefill Clustering

Running clustering during prefill hurts the most latency-sensitive phase. Offline calibration and fixed codebooks should be tested first.

### TurboQuant in Polar Space

This combines two coordinate systems without a clear advantage. PolarQuant is a valid baseline, but a hybrid should only return if experiments show polar angles beat Beta-coordinate scalar quantization at the same budget.

### PolarQuant Angle Quantization of QJL Residual

The residual is small and QJL sign bits are already designed for 1-bit inner-product correction. Polarizing the residual adds overhead without a clear statistical structure to exploit.

### Block-Diagonal or Banded Rotation

This weakens global mixing, which is the point of TurboQuant's rotation. The only acceptable block structure is P3.5, where a global fast rotation happens first.

### End-to-End Differentiable Quantization / PTQ

This changes the project category from frozen-model quantization algorithms to model adaptation. Keep out of scope for now.

### Learned Codebook from SVD of Weights

The SVD of projection weights does not necessarily decorrelate the resulting key/value activations. Calibration on actual KV tensors is the better route.

### Full RoPE-Commuting Rotation Design

A mathematically exact preconditioner that commutes with RoPE is still a deeper research problem. The active version is P3.6: test practical rotation placement and grouped-head variants.

### Progressive Embedded Codebooks

Nested refinement codebooks are operationally attractive but hard to design and awkward for streaming caches. Defer until a concrete nested Lloyd-Max construction exists.

---

## Recommended Experiment Pipeline

1. **Reference implementation** — Implement TurboQuant MSE/product, QJL, exact-Beta codebooks, and deterministic tests.
2. **P0.1/P0.2** — Validate dense Haar vs structured rotations before relying on Hadamard variants.
3. **P0.3** — Compare exact finite-d codebooks against empirical codebooks under the selected rotation.
4. **P0.4** — Add real KV-cache capture and report logit/output errors, not only vector MSE.
5. **P1.1/P1.6** — Build mixed mode and mixed precision policies from measured layer/head errors.
6. **P1.2** — Split key/value objectives and test key-favored budgets.
7. **P1.3** — Quantize norm/radius/residual-norm side information.
8. **P1.4/P1.5** — Add outlier-aware channel permutation and tiny high-precision protected-token pools.
9. **P2.2/P2.4** — Tune residual QJL structure and sketch dimension.
10. **P2.1** — Implement fused quantized attention only after the quantized representation stabilizes.

The goal of the first cycle is to reject weak assumptions quickly: if structured rotations do not match Haar quality, improve the rotation; if MSE bias matters even in shallow layers, use product/debiasing; if side information dominates the true memory ratio, compress metadata before building kernels.
