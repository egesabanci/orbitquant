# TurboQuant Optimization Proposals

> **Updated 2026-05-30** — Re-audited against TurboQuant, QJL, PolarQuant, and newer KV-cache work including RotateKV, OSCAR, OCTOPUS, AQUA-KV, outlier-token tracing, and key-favored mixed precision. The main shift is stricter measurement first, then cheap policy/debiasing changes, then larger rotation, triplet-codec, and predictive-residual branches.

Recent external signals incorporated:

- [RotateKV](https://arxiv.org/abs/2501.16383): outlier-aware channel reordering, pre-RoPE grouped-head rotation, and attention-sink-aware quantization.
- [OSCAR](https://arxiv.org/abs/2605.17757): attention-aware covariance rotations, calibrated clipping, sink/recent protection, and fused INT2 serving layout.
- [OCTOPUS](https://arxiv.org/abs/2605.21226): triplet-wise octahedral direction plus norm quantization as a stronger rotation-preconditioned codec.
- [AQUA-KV](https://arxiv.org/abs/2501.19392): inter-layer predictive residual coding before quantization.
- [More for Keys, Less for Values](https://arxiv.org/abs/2502.15075): key-favored bit allocation and finer key quantization granularity.
- [Outlier-token tracing](https://arxiv.org/abs/2505.10938): small anomalous token pools can dominate quantization quality and should be protected or excluded.

---

## P0 — Foundational: Implement First

These items are needed before the rest of the backlog can be evaluated honestly. The immediate risk is not lack of ideas; it is accepting a structured-transform or low-bit trick before measuring whether it preserves the assumptions TurboQuant relies on.

### P0.1 — Reference Implementation and Rotation Validation Harness

Build a pure PyTorch/CPU reference implementation of `TurboQuant_mse`, `TurboQuant_prod`, and QJL, then add a validation harness for rotations and codebooks.

The harness should compare dense Haar random rotation, dense Gaussian-QR rotation, Hadamard/sign-flip variants, outlier-aware permutations, pre-RoPE grouped-head variants, OSCAR-style covariance rotations, and no rotation on:

- coordinate distribution fit against the finite-d Beta law;
- coordinate covariance and higher-order dependence;
- MSE distortion for `b = 1..4`;
- inner-product bias and variance;
- attention-logit error on real KV tensors;
- softmax KL drift and attention-output error;
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
- softmax KL divergence against full precision;
- attention output error;
- hidden-state error;
- generated-token quality on a tiny deterministic task set;
- memory ratio including side information.

Synthetic vector tests are necessary but insufficient. Several proposals optimize MSE while the real failure mode may be attention-logit drift or value aggregation error.

### P0.5 — True Byte Accounting and Deployability Score

Every experiment should report paid bytes per token, not nominal bits per coordinate. Count quantized payloads, packed-bit padding, norms, radii, residual norms, clip scales, zero points, codebook IDs, protected-token pools, predictor weights, and layout metadata.

Also assign a deployability score before promoting a representation: whether it can fit a paged KV cache, whether dequantization can happen in registers, whether it supports fused attention, whether query batching is possible, and whether it avoids materializing dequantized K/V.

> **Why P0:** Several methods look equivalent at nominal bits but diverge once side information, residual windows, uncompressed sinks, and kernel friction are counted.

---

## P1 — High Impact, Low Implementation Risk

These should come after P0 because they use the same reference implementation and evaluation harness. They are mostly policy and calibration changes, not kernel work.

### P1.1 — Logit-Error Mixed Mode Across Layers and Heads

Use `TurboQuant_mse` where it is empirically safe, analytically debiased MSE where a cheap correction is enough, and `TurboQuant_prod` where residual QJL is actually needed. The decision rule should be based on measured attention-logit bias/error and softmax KL per layer and head, not only on outlier presence.

The earlier version of this idea overclaimed that shallow layers without outliers do not need unbiased correction. That is too strong: `TurboQuant_mse` can be biased even without visible outliers, especially at low bit-width. The better test is direct logit error under cached queries.

> **Paper grounding:** TurboQuant Section 3.2 proves MSE-optimal quantizers are biased for inner products. QJL Figure 2 shows outliers are layer-dependent. Combining both suggests a measured mixed-mode policy rather than a fixed all-layer choice.

### P1.2 — Separate Key/Value Objectives with Key-Favored Bit Budgets

Keys and values need different objectives. Keys control attention logits, so key quantization should prioritize low inner-product error and low bias. Values are averaged by attention weights, so they need good reconstruction under the actual attention distribution.

Start with `TurboQuant_prod` or debiased MSE for keys and `TurboQuant_mse` or attention-weighted MSE for values. Also test asymmetric bit budgets such as `K=4, V=2` against the reverse allocation at the same average KV size. Include key-only rotation, value-only rotation, and both-rotation ablations, because recent evidence suggests key rotations often buy most of the quality.

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

Keep a tiny subset of tokens at higher precision: attention sinks, system/prefix tokens, delimiter-like tokens, very recent tokens, and dynamically detected anomalous tokens. The protected pool should be tiny enough that memory ratio barely moves.

This is a narrower and more useful version of age/role-aware precision. It should be implemented as a cache-write policy with a fixed high-precision quota, then tested against uniform quantization at the same memory budget.

Do not define anomalous tokens only as high-norm tokens. Outlier-token tracing reports cases where low-magnitude key tokens disrupt channel-wise quantization, so the detector should test multiple scores: key norm, per-channel quantization range impact, attention mass, and reconstruction/output sensitivity.

> **External grounding:** [RotateKV](https://arxiv.org/abs/2501.16383) uses attention-sink-aware quantization, and [outlier-token tracing work](https://arxiv.org/abs/2505.10938) reports that a small subset of unusual tokens can dominate KV quantization error.

### P1.6 — Adaptive Bit Allocation Across Layers and Heads

Once P1.1 has per-layer/per-head error measurements, allocate bits where they buy the most quality: deeper or fragile layers, sensitive heads, and outlier-heavy heads get more bits; easy layers/heads get fewer.

The total bit budget stays fixed. The experiment is a rate-distortion scheduler over existing quantizers rather than a new quantizer.

> **Relationship to P1.1:** P1.1 chooses MSE vs product mode. P1.6 generalizes that into a mixed-precision policy.

### P1.7 — Analytical Debiasing for TurboQuant MSE

`TurboQuant_mse` is biased for inner-product estimation at low bit-width. Precompute codebook-specific scalar correction factors, or a small per-bucket correction table, to reduce score bias during attention without storing residual QJL bits.

This is attractive for latency-sensitive paths where `TurboQuant_prod` is too expensive. For `b = 1`, TurboQuant's own analysis shows the MSE estimator shrinks inner products by roughly `2/pi` in high dimensions, which makes a multiplicative correction an obvious first baseline. For higher bit-widths, estimate the correction from the exact finite-d codebook and validate on real query/key tensors.

Judge this by attention-logit bias, softmax KL, and downstream quality, not just synthetic inner-product bias.

### P1.8 — Norm-Preserving Reconstruction

After dequantizing a direction-like representation, renormalize the reconstruction to the stored or quantized original norm before scoring. This is cheap for TurboQuant-style normalized inputs, QJL residual corrections, PolarQuant radii, and OCTOPUS-style norm/direction states.

The tradeoff is that renormalization can slightly worsen raw coordinate MSE while improving logits by preserving the key norm that attention actually sees. Test both MSE and logit/output metrics.

---

## P2 — Strong Candidates, Moderate Effort

These are worth testing after the reference path is stable. They either require additional math checks, calibration data, or systems work.

### P2.1 — Fused Quantized Attention CUDA/Triton Kernel

Do not materialize dequantized K/V tensors. Compute `softmax(QK^T)V` in a fused path that reads packed quantized codes, reconstructs centroid values in registers, and streams through attention.

This is where memory compression becomes serving speed. The reference implementation should first prove quality; the fused kernel then proves deployability.

> **Prerequisite:** P0.2/P0.5 should settle the representation because the kernel needs to know whether it must apply dense rotation, FWHT, octahedral decode, residual QJL correction, predictive reconstruction, or pre-rotated-cache scoring.

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

Replace the vague "learned deterministic rotation" idea with a concrete calibration method: estimate query-side covariance for keys, score-weighted value covariance for values, and attention-weighted error sensitivity offline, then derive fixed rotations and clipping thresholds per layer/head.

This keeps serving-time cost predictable while allowing the transform to align with actual model statistics rather than worst-case spherical inputs.

The first concrete variant should be `U * H * P`: an eigenbasis from the attention-aware covariance target, followed by a normalized Hadamard transform and a balancing permutation. Compare against raw `K^T K`/`V^T V` PCA targets, pure Hadamard, random orthogonal, and no rotation.

> **External grounding:** [OSCAR](https://arxiv.org/abs/2605.17757) estimates attention-aware covariance structures offline and uses them to derive deployable rotations and clipping thresholds for low-bit KV-cache quantization.

### P2.6 — Importance-Weighted Coordinate Bit Allocation

After rotation, coordinates are close to independent in the TurboQuant story, but they may not be equally important for real attention. Estimate per-coordinate sensitivity from calibration queries and allocate more bits to coordinates that drive logit or output error.

This should be tested after P0.2/P1.4 because the optimal coordinate budget depends on the chosen rotation and channel permutation.

> **Paper grounding:** TurboQuant's scalar quantization is separable after rotation. If real rotated coordinates have unequal empirical variance or sensitivity, separable mixed precision is the natural extension.

### P2.7 — Clipped or Companded Lloyd-Max Codebooks

Structured rotations and real KV tensors may have heavier tails than the exact Beta model. Test low-bit codebooks with calibrated clipping or companding before Lloyd-Max quantization.

This is especially relevant for `b = 1` and `b = 2`, where a few tail coordinates can dominate error. The test should compare exact-Beta codebooks, empirical unclipped codebooks, clipped empirical codebooks, and log/µ-law style companding.

> **Relationship to P0.3:** P0.3 is the clean theoretical codebook. P2.7 is the practical fallback when the measured distribution does not match the theoretical one.

### P2.8 — OCTOPUS-Style Triplet Direction + Norm Codec

Test a small-block codec after rotation: normalize the vector, split rotated coordinates into contiguous triplets, store each triplet's norm, map each triplet direction on `S^2` to two octahedral coordinates, and Lloyd-Max quantize the norm and the two direction coordinates with non-uniform bit allocation.

This is the most concrete replacement for the currently deferred "joint vector quantization" idea. It avoids high-dimensional codebook explosion while exploiting local 3D geometry that independent scalar Lloyd-Max ignores. It should be tested with and without residual QJL, and against TurboQuant MSE/product and PolarQuant at matched true bytes/token.

> **External grounding:** [OCTOPUS](https://arxiv.org/abs/2605.21226) reports that triplet-wise octahedral direction quantization improves rotation-preconditioned KV codecs, especially at `b = 2` and `b = 3`.

### P2.9 — Joint Rounding for Small-Block Codecs

For OCTOPUS-style or other small-block codecs, do not independently round every scalar code. Enumerate a tiny local candidate set around the nearest direction/norm centroids and choose the tuple that minimizes reconstructed block error or score-weighted block error.

This changes only the encoder, not the decoder or bitstream, so it is a good moderate-effort experiment once a small-block codec exists.

### P2.10 — Predictive Residual KV Coding

Train compact linear predictors that reconstruct the current layer's keys from previous-layer reconstructed keys, and values from previous-layer reconstructed values plus current reconstructed keys. Store and quantize only the residual.

This is less data-oblivious than TurboQuant, but it is compatible with TurboQuant, OCTOPUS, HIGGS-style, or scalar quantizers as the residual codec. The first test should be offline calibration with fixed predictors, no online adaptation, and strict accounting for predictor weights and decode-time arithmetic.

> **External grounding:** [AQUA-KV](https://arxiv.org/abs/2501.19392) shows that inter-layer KV dependencies can make residuals substantially easier to quantize at 2-2.5 bits.

---

## P3 — Worth Testing Later

These are promising but should not block the first implementation cycle.

### P3.1 — Full Token-Prediction Objective

Move beyond reporting softmax KL as a metric and directly optimize an approximate downstream objective: KL divergence between full-precision and quantized next-token distributions, or a first-order surrogate over attention and MLP output changes.

A first-order softmax expansion can yield an attention-weighted inner-product metric. This may change which quantizer looks best, especially for values, protected-token policies, and score-correction methods.

### P3.2 — Learned Offline Quantizer Selection Policy

After enough P0-P2 experiments exist, train a simple offline policy that chooses quantizer family, bit-width, residual-QJL size, rotation type, and protected-token quota per layer/head under a fixed byte budget.

Keep the policy deployable: decision-tree or table lookup over calibration metrics, not a runtime controller that changes live cache layout during generation.

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

### Full High-Dimensional Vector Quantization on Rotated Coordinates

The random rotation's purpose is to make coordinates nearly independent. If that works, full high-dimensional joint quantization has little theoretical benefit and large codebook growth.

This no longer rules out small fixed-block codecs. P2.8 is the active version: triplet-wise direction/norm quantization has a concrete codec, tiny codebooks, and fused-kernel path.

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

1. **Reference implementation** — Implement TurboQuant MSE/product, QJL, exact-Beta codebooks, deterministic tests, and true byte accounting.
2. **P0.1/P0.2/P0.3** — Validate dense Haar vs structured rotations and exact-Beta vs empirical codebooks before relying on Hadamard variants.
3. **P0.4/P0.5** — Add real KV-cache capture and report logit bias, softmax KL, attention-output error, generated-token quality, latency, and true bytes/token.
4. **P1.7/P1.8** — Test analytical MSE debiasing and norm-preserving reconstruction as cheap corrections.
5. **P1.1/P1.2/P1.6** — Build mixed mode, key/value asymmetric objectives, key-favored budgets, and layer/head mixed precision from measured errors.
6. **P1.3/P1.4/P1.5** — Compress side information, add outlier-aware channel permutation, and add tiny high-precision protected-token pools.
7. **P2.5/P2.7** — Test attention-aware covariance rotations and calibrated clipping/companding against pure random/structured rotations.
8. **P2.8/P2.9** — Add OCTOPUS-style triplet direction/norm codec and joint rounding as the first serious non-scalar codec branch.
9. **P2.2/P2.3/P2.4** — Tune residual QJL structure, orthogonalization, and sketch dimension only where measured score errors justify it.
10. **P2.10** — Test predictive residual coding once the standalone quantizer baselines are stable.
11. **P2.1/P3.3/P3.4** — Implement fused quantized attention and query batching only after the quantized representation stabilizes.

The goal of the first cycle is to reject weak assumptions quickly: if structured rotations do not match Haar quality, improve the rotation; if MSE bias matters even in shallow layers, use product/debiasing; if side information dominates the true memory ratio, compress metadata before building kernels; if OCTOPUS-style triplets beat scalar Lloyd-Max at low bits, treat scalar TurboQuant as the baseline rather than the endpoint.
