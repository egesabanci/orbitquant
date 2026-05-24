# TurboQuant Optimization Proposals

> **Updated 2026-05-24** — Reorganized after deep analysis of all three papers (TurboQuant, QJL, PolarQuant). Proposals are now grouped by priority tier with rationale grounded in paper specifics (Beta coordinate distribution, 2.7× optimality gap, QJL Figure 2 outlier patterns, Lemma guarantees, etc.). A "Deferred / Not Recommended" section at the bottom explains why certain ideas were cut or postponed. New proposals discovered during the analysis are marked **[NEW]**.

---

## P0 — Foundational: Implement First

These are the highest-leverage, lowest-ambiguity proposals. They either unlock everything else or deliver outsized gains for near-zero risk. Start here.

### P0.1 — Structured Fast Rotations (Hadamard + Diagonal Sign Flip)

Right now, TurboQuant's biggest practical bottleneck is the **O(d²) dense random rotation** Π·x (a d×d matrix multiply). But you don't *actually* need a uniform random rotation — you just need enough mixing to decorrelate coordinates. A **Hadamard + random diagonal sign flip** sequence (HD³) costs **O(d log d)**, is perfectly GPU-butterfly-friendly, and is a known approximate JL transform. The Beta concentration and near-independence properties from TurboQuant Lemma 1 should still hold asymptotically. This one change makes TurboQuant go from "theoretically fast" to "actually fast at serving scale."

> **Paper grounding:** TurboQuant Lemma 1 — coordinate distribution after rotation is Beta(d/2, (d-1)/2). The key requirement is uniform distribution on S^{d-1}, which Hadamard + sign flip achieves (it's an orthogonal transform). The near-independence of coordinates in high dimensions [55] holds for any orthogonal matrix applied to an arbitrary input, not just random rotations. What changes is the degree of mixing, but for d ≥ 64 the concentration effects dominate.

### P0.2 — Exact Finite-Dimension Codebooks

The theoretical analysis in TurboQuant leans on high-dimensional Gaussian approximations (`N(0, 1/d)`), but practical head dimensions are commonly 64 or 128. Precompute Lloyd-Max codebooks using the **exact finite-d Beta coordinate distribution** from Lemma 1:

```
f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^{(d-3)/2}
```

for each actual dimension used by the model. This is a one-time offline computation with zero algorithmic change — just swap the codebook table. Should measurably improve low-bit distortion at b=1–4 where the Gaussian approximation is weakest.

> **Paper grounding:** TurboQuant Theorem 1 — the distortion bound D_mse = d · C(f_X, b) depends directly on how well the Lloyd-Max codebook fits the true f_X. The paper already notes that for b=1,2 the Gaussian-approximated centroids are `±√(2/π)/√d` and `{±0.453/√d, ±1.51/√d}`. Exact codebooks for d=64,128 will tighten these values.

### P0.3 [NEW] — Layer-Aware Mixed Mode: MSE for Shallow, Product for Deep

The QJL paper's Figure 2 is striking: shallow layers have **zero outliers**, deep layers have heavy outliers concentrated in ~4 fixed channels. This directly motivates a split strategy: use `TurboQuant_mse` (cheaper, no residual QJL overhead) for shallow layers where inner-product bias from outliers doesn't exist, and `TurboQuant_prod` only for deep layers where unbiased inner-product estimation actually matters.

This is a simpler, more principled version of proposal P1.1 (adaptive bit allocation). Instead of tuning continuous bit budgets per layer, it makes one binary decision per layer based on a well-documented empirical fact. The QJL paper's data gives us the decision rule for free.

> **Paper grounding:** QJL Figure 2 — "In the initial layers, no significant outlier patterns are observed. However, in the deeper layers, a few channels (approximately four) exhibit visibly larger magnitudes." TurboQuant Theorem 2 — the inner-product distortion bound scales with ‖y‖²/d · 1/4^b, so for layers without outliers the residual QJL stage is paying a bit budget cost for a problem that doesn't exist.

---

## P1 — High Impact, Low Implementation Risk

These don't change the core algorithm and deliver strong practical gains. Implement after the P0 items.

### P1.1 — Adaptive Bit Allocation Across Layers and Heads

Not all layers and heads are created equal. The QJL paper's Figure 2 shows early layers have **no outliers**, deep layers have **heavy outliers**. Rather than spending the same bit budget everywhere, allocate bits **adaptively**: more bits to deeper layers, fewer to shallow ones; more bits to critical attention heads, fewer to redundant ones. The total bit budget stays the same, but the *rate-distortion* improves dramatically because you're spending precision where the model is actually sensitive.

> **Note:** This is the continuous-budget generalization of P0.3. P0.3 gives a simple binary split as a quick win; P1.1 tunes the exact allocation once the infrastructure exists.

### P1.2 — Separate Key and Value Quantization Objectives

Keys and values serve fundamentally different roles in attention. Keys determine the attention distribution via inner products with the query — they need unbiased inner-product estimation (TurboQuant_prod). Values are aggregated via attention-weighted summation — they need good reconstruction where attention mass is high, but unbiasedness is irrelevant. Use `TurboQuant_prod` for keys and an MSE or attention-weighted MSE quantizer for values.

This is a clean theoretical insight that costs nothing to implement and should improve quality at the same average KV size, because value errors only matter in proportion to attention mass.

> **Paper grounding:** TurboQuant defines two distortion measures — D_mse (Eq. 1) and D_prod (Eq. 2). The paper applies one or the other uniformly. But §1.1 notes that for values, "a standard token-wise quantization is very effective and efficient in practice" (QJL §3.2). The key insight: keys need Theorem 2's unbiased guarantee; values only need Theorem 1's MSE bound.

### P1.3 — Quantized Norm and Radius Storage

The papers remove scale/zero-point overhead for coordinates, but still store scalar side information: QJL stores key norms ‖kᵢ‖₂, PolarQuant stores radii, and TurboQuant_prod stores residual norms. These scalars are small but **persistent overhead** at long context — one float per token per head. At sequence length 128K with 32 heads, that's 128K × 32 × 4 bytes = 16 MB in FP32, or 8 MB in FP16. It's pure overhead.

Quantize norms in **log space** with a tiny shared format (e.g., 8-bit log-scale with head-specific bias/scale). The dynamic range of KV norms is typically narrow (< 10×), so 8 bits is more than sufficient. This directly improves real compression ratio without touching the core vector quantizer.

> **Paper grounding:** QJL §3.1 — "store the quantized vector k̃ᵢ and the key norm νᵢ in the cache." TurboQuant §1.1 — "we can compute and store the L2 norms in floating-point precision and rescale." Both papers acknowledge the overhead but don't optimize it.

### P1.4 [NEW] — Entropy Coding Layer on Quantized Indices

After scalar quantization, adjacent coordinates in the rotated space often land in the same quantization bucket, especially near-zero coordinates. For b=1–2 on near-Gaussian coordinates, run-length encoding (RLE) the index stream can save 10–30% additional space.

This is independent of the quantization algorithm — it's a pure entropy coding layer applied after `Quant_mse`. It has zero impact on distortion because it's lossless on the already-quantized indices. The only cost is a small decode pass during dequantization, which is trivially fast.

> **Relationship to P0.2:** Exact finite-d codebooks produce tighter clusters around centroids, which increases the probability of adjacent indices being identical, making RLE even more effective. These two proposals compound.

---

## P2 — Solid Improvements, Moderate Implementation Effort

These have strong theoretical backing and clear paths to implementation, but require more engineering or have slightly more uncertainty.

### P2.1 — Dithered / Stochastic Quantization

Add controlled pseudo-random noise before rounding to break deterministic quantization artifacts. Dithering trades a small variance increase for eliminating systematic bias patterns that accumulate over many attention heads. This is standard in audio ADCs but unexplored in KV cache quantization.

**Subtractive dithering** is the key variant: add noise before quantization, then subtract the same noise after dequantization. This recovers the variance cost — the only penalty is a small increase in quantization step size. For many-head attention (32–128 heads), the systematic bias from deterministic rounding compounds, while the variance from dithering averages out. The net effect should be positive for downstream quality.

> **Paper grounding:** TurboQuant Theorem 1 proves D_mse for the deterministic quantizer. With subtractive dithering, the expected MSE remains the same but the error becomes signal-independent (white noise), which is perceptually superior and less harmful to softmax operations.

### P2.2 — Fused Quantized Attention CUDA Kernel

Don't materialize the dequantized matrix at all. Compute `softmax(Q·Kᵀ)V` in a single fused CUDA kernel that reads quantized codes, looks up centroids on-the-fly in registers, and streams through the computation. This eliminates the memory round-trip of the full dequantized KV cache — the same principle that makes AWQ/GPTQ fast for weight quantization. The quantized codes stay compressed in GPU memory until the moment they're needed in registers.

> **Prerequisite:** P0.1 (Hadamard rotation) should be implemented first, since the fused kernel needs to know the transform structure. With Hadamard, the forward/inverse transforms are FFT-like butterfly operations that fuse naturally into the attention kernel.

### P2.3 — Importance-Weighted Per-Coordinate Bit Allocation

P1.1 allocates bits per layer and head. Go finer: allocate bits per **coordinate** within the rotated space. After rotation, some coordinate directions contribute far more to attention score variance than others. Precompute per-coordinate sensitivity by measuring how perturbation along each rotated basis direction affects the softmax output, then allocate bits proportional to sensitivity.

This is a natural extension: after rotation, coordinates are near-independent, so the optimal bit allocation is separable — allocate more bits to high-variance coordinates. The sensitivity can be estimated offline from a calibration set (like P2.8) or derived analytically from the Beta distribution shape.

> **Paper grounding:** TurboQuant Lemma 1 — coordinates follow Beta(d/2, (d-1)/2), which is symmetric around 0 but has variance ~1/d. All coordinates have equal variance asymptotically, but for finite d (64, 128) and real (non-spherical) inputs, the empirical variance per rotated coordinate will differ. Allocating bits proportionally to log-variance is the rate-distortion optimal strategy for independent Gaussian-like sources.

### P2.4 — Orthogonalized Residual Sketches

QJL explicitly reports empirical gains from orthogonalizing JL rows (QJL §4.1: "We observed that orthogonalized random Gaussian matrices consistently exhibit better practical performance... we first generate a random JL matrix S with i.i.d. Gaussian entries and then orthogonalize its rows using QR decomposition"). Apply this same idea specifically to TurboQuant's residual QJL stage.

The residual sketch matrix S in TurboQuant_prod is currently i.i.d. Gaussian. Replacing it with an orthogonalized version reduces estimator variance at zero additional memory cost — the sketch dimension and bit budget are unchanged.

> **Paper grounding:** QJL §4.1. TurboQuant Definition 1 (QJL) — the sketch matrix S is defined as i.i.d. N(0,1). The orthogonalized variant has the same unbiasedness guarantee (Lemma 4) but lower variance because the rows are decorrelated. This pairs naturally with P2.5 (variable-size residual).

### P2.5 — Variable-Size Residual QJL

TurboQuant_prod currently spends exactly one additional bit per coordinate on the residual QJL stage. Generalize: let the residual sketch dimension `m` be a free parameter, then optimize the split `(b_mse, m)` under a fixed total bit budget. For easy layers or small residuals, `m < d` may preserve unbiasedness while saving memory. For fragile layers, `m > d` may outperform adding another scalar bit everywhere.

The optimization problem: given total budget B = b_mse · d + m bits, find the split that minimizes D_prod from Theorem 2.

> **Paper grounding:** TurboQuant Theorem 2 — D_prod ≤ √(3π)/2 · ‖y‖²/d · 1/4^{b_mse} · (something dependent on m/d). The current construction fixes m = d, but the bound suggests a tradeoff surface worth exploring.

### P2.6 — Learned Deterministic Rotations

Random rotation works for *any* input, but it's pessimistic — it treats your data as worst-case. In practice, LLM key embeddings within a given layer aren't worst-case; they have structure. Do a **cheap offline calibration** (a few forward passes on representative data) and learn a **deterministic per-layer rotation** that *maximally concentrates* coordinates into low-entropy distributions, so your scalar codebook becomes much better matched to the real data.

You stay data-oblivious at *serving time* (no per-token adaptation), but the rotation you use is no longer random — it's optimized. This could meaningfully close the 2.7× gap to Shannon's bound, especially at practical bit-widths like b=2–4.

> **Prerequisite:** P0.1 (Hadamard) should be done first as the baseline. P2.6 is then an enhancement: start with Hadamard + diagonal sign flip as the initialization, then optimize the diagonal signs via gradient descent to minimize distortion on calibration data. The resulting rotation is still O(d log d).

---

## P3 — Worth Testing Later

These are promising but more speculative, or require infrastructure from earlier tiers to be built first.

### P3.1 — Distortion Metric Aligned with Token Prediction

The current analysis minimizes MSE and inner product error. But what actually matters for LLMs is **KL divergence in the output token distribution**. Derive a quantization scheme that directly minimizes expected KL divergence between softmax outputs with and without quantization.

Even a first-order Taylor expansion of the softmax yields a weighted inner-product distortion metric that's better aligned than raw MSE. The weights are query-dependent: coordinates that strongly influence high-attention tokens get higher weight. This could change which quantizer variant is optimal and how we evaluate success.

> **Paper grounding:** TurboQuant §1.1 — the distortion measures D_mse and D_prod are proxies. The true objective is downstream task quality (§4.2–4.3). P3.1 makes the proxy more faithful to the objective, which may reveal that certain proposals (P1.2, P1.1) matter more than currently suspected.

### P3.2 — Analytical Debiasing for TurboQuant MSE

TurboQuant_mse is excellent for reconstruction, but its dequantized vectors are biased for inner product estimation at low bit-widths. Instead of always spending a full residual QJL stage, precompute a **codebook-specific scalar debiasing factor** (or a small per-bucket correction table) so that `⟨q, DeQuant_mse(k)⟩` is closer to unbiased.

The 1-bit case already has a clean multiplicative bias; higher bit-widths should have measurable calibration curves from the Lloyd-Max centroids and Beta coordinate distribution. This changes only dequantization/scoring, not the encoded representation, making it cheap to test.

> **Note:** This is partially subsumed by P0.3 (layer-aware mixed mode) and the existence of TurboQuant_prod. The value here is mainly for applications where two-stage quantization is undesirable for latency reasons.

### P3.3 — Age- and Role-Aware KV Precision

Treat KV tokens differently based on their role in generation. Recent generated tokens, system/prefix tokens, delimiter tokens, and attention sinks often deserve higher precision than old bulk context tokens. This is a simple static serving policy that can be decided at cache write time and should be easy to combine with TurboQuant's streaming setup.

> **Relationship to P1.1:** P1.1 allocates bits spatially (across layers/heads); P3.3 allocates temporally (across sequence positions). They compose naturally.

### P3.4 [NEW] — Residual QJL Coordinate Subsampling

Instead of applying QJL to all `d` coordinates of the residual, randomly subsample `m < d` coordinates and only store sign bits for those. The inner product estimator needs adjustment (importance sampling reweighting), but this directly reduces the residual stage memory by factor `d/m`.

This is a more aggressive version of P2.5. Where P2.5 treats `m` as a free parameter, P3.4 explicitly uses random subsampling, which has known variance properties from the importance sampling literature.

> **Paper grounding:** TurboQuant Lemma 4 — the QJL estimator is an average of `d` i.i.d. samples zᵢ = √(π/2) · sᵢᵀy · sign(sᵢᵀx). Subsampling to `m` coordinates gives variance proportional to 1/m instead of 1/d. For residuals with small ‖x‖, the variance increase may be acceptable for the memory savings.

### P3.5 [NEW] — Lookup-Table Quantized Attention (b=1,2)

For b=1,2 bit quantization, all possible inner product contributions from quantized codes are enumerable. With b=1, there are only 2 centroids per coordinate; with b=2, there are 4. Precompute a lookup table mapping `(quantized_code_index, query_coordinate_value)` → contribution to inner product.

At inference time, the attention computation becomes pure accumulation from the LUT — no floating-point multiplies. For b=1 with d=128, the LUT has only 128×2 = 256 entries per query, fitting comfortably in L1 cache. This would be absurdly fast and pairs naturally with P2.2 (fused kernel).

> **Limitation:** Only practical for b ≤ 2. For b=3 (8 centroids), the LUT grows to 128×8 = 1024 entries — still small. For b=4 (16 centroids), it's 2048 entries. Beyond that, direct multiply-accumulate is cheaper.

### P3.6 [NEW] — Block-Structured Hybrid Rotation (Hadamard + Small Dense Blocks)

Use Hadamard for the O(d log d) bulk rotation, then apply **small** dense random rotations (e.g., 8×8 blocks) on top for residual mixing. Cost: O(d log d + d·8²) = O(d log d), still sub-quadratic.

The Hadamard transform provides d/2 two-coordinate mixing operations. Adding small dense blocks gives within-group mixing that Hadamard's butterfly structure may miss. The result should achieve better decorrelation than pure Hadamard while staying much cheaper than full dense rotation.

> **Relationship to P0.1:** P0.1 is the baseline. P3.6 is an enhancement if empirical decorrelation from pure Hadamard proves insufficient. The "HD³" construction in P0.1 (three rounds of Hadamard + sign flip) is already quite strong; this adds a final mixing layer.

### P3.7 [NEW] — Online Adaptive Bit-Width via Streaming Variance Tracking

P1.1 and P2.3 allocate bit budgets statically based on offline calibration. But the distribution of rotated coordinates may shift between prefill and decode phases, or across different prompts. Maintain an exponentially weighted moving average (EWMA) of per-coordinate variance during generation and reallocate bits on-the-fly.

This is simple to implement: track a running variance estimate per rotated coordinate, periodically recompute the optimal bit allocation (which is just sorting by variance and assigning bits proportionally), and update the per-coordinate bit-width for the next block of tokens. No model changes needed.

> **Advantage over static allocation:** Handles non-stationarity. If a particular prompt type or generation phase (e.g., chain-of-thought) produces different embedding statistics, the bit allocation adapts automatically.

### P3.8 [NEW] — Norm-Baked Scalar Quantizer (Eliminate Norm Storage)

Instead of normalizing vectors to the unit sphere (storing norms separately, as in P1.3), bake the norm into the quantization decision. For a vector with L2 norm `r`, the rotated coordinates are scaled by `r`: each coordinate follows a Beta distribution scaled to `[-r, r]`.

Precompute Lloyd-Max codebooks for a grid of `r` values, then interpolate at runtime. The quantized representation stores only code indices — the norm is implicitly encoded in which codebook was used. This eliminates separate norm storage entirely.

> **Tradeoff:** Requires per-vector codebook adaptation (cheap if precomputed and interpolated) and slightly larger codebook storage. The win is that every stored bit goes toward vector data. Pairs with P1.3 — if P3.8 works, P1.3 becomes unnecessary.

### P3.9 [NEW] — Query Batching Optimization for Fused Kernels

Extension of P2.2. In serving, multiple queries often hit the same KV cache simultaneously (batch inference). Structure the fused kernel to load quantized codes once from GPU memory, then compute attention for multiple queries in a single pass.

This compounds the memory bandwidth savings from quantization: with batch size B, the effective bandwidth per query is divided by B. For b=2 quantization (4× compression) with batch size 8, the effective memory traffic per query is 32× lower than FP16 baseline.

> **Prerequisite:** P2.2 (fused kernel). This is a batched generalization of the single-query fused attention kernel.

### P3.10 [NEW] — Runtime Codebook Refinement via Streaming k-Means

The theoretical Beta codebook (P0.2) is computed offline. At serving time, the actual rotated coordinate distribution may deviate slightly from the theoretical Beta due to finite dimensions, non-spherical inputs, or distribution shift. Run a lightweight streaming k-means (1–2 Lloyd iterations) on actual rotated coordinates to refine centroids.

This stays online but uses runtime data. Unlike the rejected proposal (see Deferred), it does **not** add latency to the prefill phase — the refinement can happen asynchronously in background, updating centroids for future tokens.

> **Paper grounding:** TurboQuant Algorithm 1 — the codebook is a global parameter computed once. This adds an optional online refinement step that converges toward the empirical distribution while staying within the same algorithmic framework.

### P3.11 [NEW] — Quantized Rotation Matrix Storage

The rotation matrix Π itself is d×d floating-point numbers. For d=128, that's ~65 KB per head. Across 32 layers × 32 heads, that's ~67 MB just for rotation matrices in FP32.

Store Π in bfloat16 (halves the cost) or even int8 with per-column scale factors. The rotation only needs to approximately whiten coordinates — exact precision isn't critical. TurboQuant Theorem 1's bound depends on coordinates being Beta-distributed, which holds approximately even with quantized rotation entries.

> **Note:** If P0.1 (Hadamard) is used, this becomes largely moot since structured transforms have near-zero storage (just the diagonal sign flip bits). This is mainly relevant if dense random rotations are kept for any reason.

### P3.12 — ANN-Specific TurboQuant Scoring Path

For nearest-neighbor search, implement asymmetric scoring directly over TurboQuant codes and rerank only a small candidate set in full precision or higher precision. The TurboQuant paper already shows strong recall and near-zero indexing time; the next systems win is avoiding full dequantization during search. This would make TurboQuant more compelling as a practical vector database primitive.

---

## Deferred / Not Recommended

These proposals were evaluated and found to be either invalid, premature, out of scope, or net-negative after deeper paper analysis. They are documented here with reasons to avoid revisiting them without new evidence.

### ❌ Joint (Vector) Quantization on Rotated Coordinates (was #4)

**Why not:** The random rotation's purpose is to make coordinates near-independent (TurboQuant §3.1: "distinct coordinates of Π·x become nearly independent"). If independence is achieved, joint quantization of pairs/triplets offers zero gain over scalar quantization. The 2.7× gap to Shannon's bound comes from scalar quantization on independent coordinates, not from residual correlation. For b=2, a scalar has 4 entries; 2D joint has 16. The codebook table grows exponentially (2^{b·group_size}) with no theoretical benefit when independence holds. **Revisit only if empirical evidence shows significant residual correlation after rotation.**

### ❌ Multi-Stage Residual Quantization (3–5 stages) (was #5)

**Why not:** The 2-stage design (MSE + QJL) is already elegant and provably near-optimal. Each successive residual has exponentially less energy. The audio codec analogy (CELP, Opus) doesn't transfer — those signals have strong harmonic structure in residuals, which rotated Gaussian-ish coordinates don't. Unless you can demonstrate exploitable structure in the post-QJL residual, adding stages is unjustified complexity. **Revisit only with evidence of structured residuals after stage 2.**

### ❌ Frequency-Domain Transform (DCT/DST) (was #6)

**Why not:** Covered by P0.1 (Hadamard). Hadamard has superior theoretical JL properties and GPU butterfly support. DCT/DST don't have the same guarantees for decorrelation. Unless there's strong empirical evidence that LLM embeddings are naturally sparse in DCT basis (unlikely given the learned nature of the embeddings), this adds complexity for no clear benefit. **Consolidated into P0.1/P3.6.**

### ❌ Query-Conditional Adaptive Bit Width (was #9)

**Why not:** Operationally impractical for autoregressive generation. A "coarse pass first, refine top-k" strategy requires two scans of the KV cache per generated token. The latency cost would dominate any memory savings. This is a viable idea for offline ANN search but not for real-time token generation. **Valid for ANN use case (see P3.12), invalid for KV cache serving.**

### ❌ Asymmetric Query/Key Transformations (was #10)

**Why not:** Underspecified. "Different but related projections" needs a concrete mathematical construction before it's evaluable. The QJL asymmetry (sign-bit on key, full projection on query) is already proven optimal for b=1 via Lemma 3.2. Generalizing to b>1 with different matrices is an open research problem, not an experiment candidate. **Defer until a concrete construction is proposed with theoretical guarantees.**

### ❌ Cache-Line-Aware Code Layout (was #12)

**Why not:** Premature at the project's current stage. There is no CUDA kernel yet. Cache-line optimization is something you do after you have a working fused kernel (P2.2) and profiling data showing specific cache-miss bottlenecks. The algorithmic gains from P0.1 and P2.2 dwarf any layout tricks. **Defer until after P2.2 is implemented and profiled.**

### ❌ Cross-Head / Cross-Layer Codebook Sharing (was #13)

**Why not:** Solving the wrong problem. The codebook is tiny: for b=2, it's 4 centroid values per coordinate. Even with 128 coordinates per head and 32 heads per layer, the total codebook is ~16 KB per layer. The memory is overwhelmingly in the index array (b·d·num_heads·num_tokens bits). Saving 16 KB is irrelevant when the KV cache for 128K tokens is gigabytes. **Remove.**

### ❌ Quantization-Aware Prefill Optimization (was #14)

**Why not:** Breaks TurboQuant's key advantage of being fully data-oblivious and online. Running k-means refinement during prefill adds latency to the most latency-sensitive phase of inference. For serving, prefill speed matters. The streaming variant (P3.10) is the correct way to incorporate runtime data without blocking the critical path. **Superseded by P3.10.**

### ❌ TurboQuant in Polar Space (was #15)

**Why not:** The recursive polar transform adds O(d log d) computation that Hadamard (P0.1) already achieves with better theoretical backing. The distributions may be more concentrated (PolarQuant Lemma 2), but you lose the clean 2.7× optimality bound from TurboQuant Theorem 1. Without proving superiority over Hadamard-based TurboQuant, this is speculative hybridization. **Revisit only if empirical comparison shows polar angles yield lower distortion than Beta-coordinate scalar quantization at the same bit-width.**

### ❌ PolarQuant Angle Quantization of QJL Residual (was #16)

**Why not:** The residual after MSE quantization has small norm with no particular angular structure. The sign bit is information-theoretically optimal for 1-bit quantization (QJL Lemma 4). Converting to polar coordinates for a near-zero residual is unnecessary overhead. **Remove.**

### ❌ Block-Diagonal / Banded Rotation (was #17)

**Why not:** Directly contradicts the core insight of TurboQuant. The random rotation's job is to decorrelate all coordinates so scalar quantization works. Block-diagonal rotations preserve within-block correlations, which is exactly what you want to destroy. The result would be worse distortion for the same bit budget. **Remove.**

### ❌ End-to-End Differentiable Quantization / PTQ (was #18)

**Why not:** Requires model fine-tuning or retraining. This is a completely different category of work (model modification vs. algorithmic improvement). Out of scope for the current project phase, which focuses on quantization algorithm improvements that work with frozen models. **Out of scope.**

### ❌ Learned Codebook from SVD of Weights (was #19)

**Why not:** Needs significant implementation and model access. The SVD of weight matrices gives a rotation basis, but there's no guarantee it decorrelates key embeddings — the weight matrices transform hidden states to keys, not the keys themselves. Also breaks the data-oblivious property. **Out of scope for current phase. Partially addressed by P2.6 (learned rotations) which uses calibration data directly.**

### ❌ RoPE-Compatible Rotations (was #25)

**Why not:** Important for deployment but a hard open mathematical problem. Finding a preconditioner that commutes with RoPE's 2D rotation blocks is non-trivial. This is a research direction, not an experiment candidate. **Document as future work. Do not implement now.**

### ❌ Progressive Embedded Codebooks (was #26)

**Why not:** Operationally complex. The "append refinement bits" pattern requires a codebook family with strict nesting properties (harder to design than it sounds) and assumes you can revisit old tokens (not always true in streaming). The memory-pressure knob is appealing but the engineering cost is high. **Defer until a concrete nested codebook construction exists.**

---

## Summary: Recommended Experiment Pipeline

1. **Reference implementation** — Pure Python/PyTorch TurboQuant (MSE + Product) with dense random rotation (baseline)
2. **P0.1** — Swap dense rotation for Hadamard + sign flip. Measure speedup and distortion delta.
3. **P0.2** — Compute exact Beta(d) codebooks for d=64,128. Measure distortion improvement vs. Gaussian approximation.
4. **P0.3** — Implement layer-aware mixed mode. Profile outlier presence per layer; apply MSE-only to shallow layers.
5. **P1.1** — Add per-layer/per-head bit budget knobs on top of P0.3.
6. **P1.2** — Split key/value quantization strategies.
7. **P1.3** — Quantize norms in log space.
8. **P1.4** — Add RLE entropy coding on index streams.

After these eight steps, the core algorithmic improvements are in place. P2 items (CUDA kernel, dithering, orthogonalized residuals, learned rotations, variable-size QJL) then build on this foundation. P3 items are explored opportunistically as time and evidence permit.
