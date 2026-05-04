# TurboQuant Optimization Proposals

## 1. Structured Fast Rotations — Replace Dense with Hadamard + Diagonal

Right now, TurboQuant's biggest practical bottleneck is the **O(d²) dense random rotation** Π·x (a d×d matrix multiply). But you don't *actually* need a uniform random rotation — you just need enough mixing to decorrelate coordinates. A **Hadamard + random diagonal sign flip** sequence (HD³ or so) costs **O(d log d)**, is perfectly GPU-butterfly-friendly, and is a known approximate JL transform. The Beta concentration and near-independence properties should still hold asymptotically. This one change makes TurboQuant go from "theoretically fast" to "actually fast at serving scale."

## 2. Learned Deterministic Rotations — Close the 2.7× Gap

Random rotation works for *any* input, but it's pessimistic — it treats your data as worst-case. In practice, LLM key embeddings within a given layer aren't worst-case; they have structure. If you do a **cheap offline calibration** (a few forward passes on representative data) and learn a **deterministic per-layer rotation** that *maximally concentrates* coordinates into low-entropy distributions, your scalar codebook becomes much better matched to the real data. You stay data-oblivious at *serving time* (no per-token adaptation), but the rotation you use is no longer random — it's optimized. This could meaningfully close the 2.7× gap to Shannon's bound, especially at practical bit-widths like b=2–4. The subtlety: you're not sacrificing the online/streaming property, you're just making the *preconditioning* smarter.

## 3. Adaptive Bit Allocation Across Layers and Heads

Not all layers and heads are created equal. The QJL paper's Figure 2 shows early layers have **no outliers**, deep layers have **heavy outliers**. Rather than spending the same bit budget everywhere, allocate bits **adaptively**: more bits to deeper layers, fewer to shallow ones; more bits to critical attention heads, fewer to redundant ones. The total bit budget stays the same, but the *rate-distortion* improves dramatically because you're spending precision where the model is actually sensitive. This is subtle because it doesn't change the quantization algorithm at all — just the *budget allocation* — but it's where the biggest quality-per-bit gains live in practice.

---

## Algorithmic / Statistical Optimizations

### 4. Small-Group Joint (Vector) Quantization on Rotated Coordinates

The 2.7× gap to Shannon's bound exists because TurboQuant uses pure scalar quantization — each coordinate quantized independently. But coordinates after rotation aren't **perfectly** independent; they're only asymptotically independent. Quantizing **pairs or triplets** of adjacent coordinates jointly (2D/3D Lloyd-Max) would capture residual correlations that scalar quantization misses. For b=2 bits, a scalar codebook has 4 entries — a 2D joint codebook has only 16 (still tiny). Codebook sizes stay tractable for groups of 2–3 coordinates, directly attacking the 2.7× gap without hitting the curse of dimensionality.

### 5. Residual Vector Quantization (Multi-Stage)

Instead of just two stages (MSE quantizer + single QJL on residual), use **3–5 stages** of quantization on successive residuals, each with a small codebook. This is the core technique in audio codecs (CELP, Opus) and can achieve strictly better rate-distortion tradeoffs by spending bits where the residual still has structure. The final stage can remain QJL for unbiasedness. More stages = finer-grained control over the rate-distortion curve.

### 6. Frequency-Domain Transform Instead of Random Rotation

The Hadamard transform (proposal #1) is just one structured transform. **DCT, DST, or trainable butterfly factorizations** may achieve even better coordinate concentration for LLM embeddings specifically. The key insight: the transform doesn't need to be random — it just needs to approximately whiten coordinate distributions, and LLM embeddings may already be nearly sparse in certain bases (e.g., after DCT, most energy concentrates in low-frequency components).

### 7. Dithered / Stochastic Quantization

Add controlled pseudo-random noise before rounding to break deterministic quantization artifacts. Dithering trades a small variance increase for eliminating systematic bias patterns that accumulate over many attention heads. This is standard in audio ADCs but unexplored in KV cache quantization. Subtractive dithering (add noise before quantize, subtract after dequantize) can even recover the variance cost.

### 8. Importance-Weighted Per-Coordinate Bit Allocation Within a Layer

Proposal #3 allocates bits per layer and head. Go finer: allocate bits per **coordinate** within the rotated space. After rotation, some coordinate directions contribute far more to attention score variance than others. Precompute per-coordinate sensitivity by measuring how perturbation along each rotated basis direction affects the softmax output, then allocate bits proportional to sensitivity. Same total bit budget, smarter distribution.

### 9. Query-Conditional Adaptive Bit Width

All keys currently get the same bit-width regardless of the query. But attention scores are query-dependent — a key that gets near-zero attention doesn't need precision. Use a fast **coarse estimator** first (e.g., 1-bit QJL), identify top-k keys that receive significant attention, then allocate remaining bit budget to refine only those keys. This gives **variable-rate compression per token** — spend bits where the attention mass is.

### 10. Asymmetric Query/Key Transformations

QJL already does this (full JL on query, sign-bit on key). Generalize: use **different but related** projections for keys and queries — e.g., a learned projection for queries that's the pseudoinverse of the key projection in the inner-product sense. This could achieve strictly lower inner-product distortion for the same bit budget by aligning the projections with the interaction structure.

---

## Systems / Implementation Optimizations

### 11. Fused Quantized Attention CUDA Kernel

Don't materialize the dequantized matrix at all. Compute `softmax(Q·Kᵀ)V` in a single fused CUDA kernel that reads quantized codes, looks up centroids on-the-fly in registers, and streams through the computation. This eliminates the memory round-trip of the full dequantized KV cache — the same principle that makes AWQ/GPTQ fast for weight quantization. The quantized codes stay compressed in GPU memory until the moment they're needed in registers.

### 12. Cache-Line-Aware Layout for Quantized Codes

Reorder quantized indices in memory so codes accessed together during attention share cache lines. For example: interleave codes from the same token position across all heads, or group codes by frequently co-occurring index values. The quantized format gives complete flexibility to reorganize — you're not bound to the original embedding layout.

### 13. Cross-Head / Cross-Layer Codebook Sharing

Different attention heads often learn redundant patterns. Share the same Lloyd-Max codebook across all heads in a layer (or even across layers), with only **per-head scale factors** stored separately. This reduces total codebook storage and improves cache locality — the same centroids stay hot in cache across all head computations.

### 14. Quantization-Aware Prefill Optimization

During the prefill phase (first forward pass with full prompt), you see the actual KV cache distribution **before** generation begins. Do a lightweight one-pass k-means refinement of the theoretical codebook based on the actual prefill data. This is data-aware but cheap — only one pass, stays online, and amortizes over all subsequent generation tokens from the same prompt.

---

## Cross-Paper Hybridizations

### 15. TurboQuant in Polar Space

Apply TurboQuant's scalar quantization **after** PolarQuant's polar transformation. The angles from the recursive polar transform have analytically known distributions (PolarQuant Lemma 2) that are even more concentrated than the Beta distribution in Cartesian space. Lloyd-Max on these polar angles could achieve lower per-coordinate distortion than either paper alone.

### 16. PolarQuant Angle Quantization of QJL Residual

The QJL residual (used in TurboQuant's inner-product stage) could be quantized in polar coordinates instead of sign-bit. Since the residual has small norm, angular information may be more informative than the sign pattern. Use PolarQuant's recursive polar code on the residual for better bit efficiency in the second stage.

### 17. Spatial Locality Exploitation in the Rotation

The random rotation treats all input coordinates equally. But neighboring key dimensions within a head often have statistical dependencies. Use a **block-diagonal** or **banded** rotation matrix that mixes coordinates within local windows rather than globally, preserving structural information that pure scalar quantization alone can't capture, while still achieving sufficient mixing.

---

## Learning-Based Optimizations

### 18. End-to-End Differentiable Quantization for Post-Training

Make the quantizer differentiable (straight-through estimator or Gumbel-softmax) and run a few gradient steps **per layer** to make the model's weights tolerant to the quantization noise profile. This is post-training quantization (PTQ) applied to KV cache rather than weights — the model learns to route information through channels that survive quantization better.

### 19. Learned Codebook Initialization from Model Weights

The rotation matrix Π doesn't need to be random. Initialize it via **SVD of the layer's weight matrices** — project keys into the basis of weight singular vectors, which are already optimized for the inner products the model needs. This gives a task-specific rotation at zero calibration cost — the weights already encode what directions matter.

### 20. Distortion Metric Aligned with Token Prediction

The current analysis minimizes MSE and inner product error. But what actually matters for LLMs is **KL divergence in the output token distribution**. Derive a quantization scheme that directly minimizes expected KL divergence between softmax outputs with and without quantization. This is the true optimization target — everything else is a proxy. Even a first-order Taylor expansion of the softmax could yield a weighted inner-product distortion metric that's better aligned than raw MSE.

---

## Additional Proposals Worth Trying

### 21. Analytical Debiasing for TurboQuant MSE

`TurboQuant_mse` is already excellent for reconstruction, but its dequantized vectors are biased for inner product estimation at low bit-widths. Instead of always spending a full residual QJL stage, precompute a **codebook-specific scalar debiasing factor** (or a small per-bucket correction table) so that `<q, DeQuant_mse(k)>` is much closer to unbiased. The 1-bit case already has a clean multiplicative bias; higher bit-widths should have measurable calibration curves from the Lloyd-Max centroids and Beta coordinate distribution. This is a cheap path to test because it changes only dequantization/scoring, not the encoded representation.

### 22. Variable-Size Residual QJL

`TurboQuant_prod` currently spends exactly one additional bit per coordinate on the residual QJL stage. Generalize this to a residual sketch dimension `m` that can be smaller or larger than `d`, then optimize the split between `(b_mse, m)` under a fixed total bit budget. For easy layers or small residuals, `m < d` may preserve unbiasedness while saving memory; for fragile layers, a slightly larger residual sketch may outperform adding another scalar bit everywhere.

### 23. Quantized Norm and Radius Storage

The papers remove scale/zero-point overhead for coordinates, but still store scalar side information: QJL stores key norms, PolarQuant stores radii, and `TurboQuant_prod` stores residual norms. These scalars are small but persistent overhead at long context. Quantize them in **log space**, share them per block/head when possible, or use a tiny exponent-mantissa format tuned to observed KV norm ranges. This should be tested because it directly improves real compression ratio without touching the core vector quantizer.

### 24. Separate Key and Value Quantization Objectives

Keys and values do not need the same distortion metric. Keys should preserve logits and attention ordering, while values should preserve the attention-weighted output after softmax. Use `TurboQuant_prod`-style unbiased inner product preservation for keys, but use an MSE or attention-weighted MSE quantizer for values. This can produce better quality at the same average KV size because value errors only matter in proportion to attention mass.

### 25. RoPE-Compatible Rotations

Random rotations, Hadamard transforms, and learned rotations all add serving complexity unless they can be folded into the model or fused cheaply. For RoPE-based models, design preconditioners that **commute with RoPE's 2D rotation blocks** or operate blockwise inside RoPE coordinate pairs. If the transform can be folded into Q/K projections or applied before RoPE without changing attention semantics, TurboQuant gets much easier to deploy in existing inference stacks.

### 26. Progressive Embedded Codebooks

Build nested codebooks where a 3-bit representation extends a 2-bit representation, and a 4-bit representation extends the 3-bit one. This enables precision upgrades without re-quantizing: start all tokens at low precision, then append refinement bits for important heads, recent tokens, outliers, or high-attention candidates. It also gives a clean memory-pressure knob for serving systems.

### 27. Exact Finite-Dimension Codebooks

The analysis often leans on high-dimensional Gaussian approximations, but practical head dimensions are commonly 64 or 128, and outlier splits can reduce the effective dimension further. Precompute Lloyd-Max codebooks using the **exact finite-d Beta coordinate distribution** for each actual dimension used by the model. This is low risk and may improve low-bit distortion without changing the algorithmic structure.

### 28. Orthogonalized Residual Sketches

QJL reports empirical gains from orthogonalizing JL rows. Apply the same idea specifically to TurboQuant's residual QJL stage: use orthogonalized Gaussian sketches, SRHT-style sketches, or other structured orthogonal sketches for the residual. This is likely to reduce estimator variance with minimal conceptual change, and it pairs naturally with variable-size residual QJL.

### 29. Age- and Role-Aware KV Precision

Treat KV tokens differently based on their role in generation. Recent generated tokens, system/prefix tokens, delimiter tokens, and attention sinks often deserve higher precision than old bulk context tokens. This is distinct from query-conditional refinement: it is a simple static serving policy that can be decided at cache write time and should be easy to combine with TurboQuant's current streaming setup.

### 30. ANN-Specific TurboQuant Scoring Path

For nearest-neighbor search, implement asymmetric scoring directly over TurboQuant codes and rerank only a small candidate set in full precision or higher precision. The paper already shows strong recall and near-zero indexing time; the next systems win is avoiding full dequantization during search. This would make TurboQuant more compelling as a practical vector database primitive, not just a compression method.
