# OrbitQuant

OrbitQuant is an experimental optimization project for pushing TurboQuant-style vector quantization further. The repository collects research papers, concrete optimization proposals, and an implementation roadmap for future experiments around LLM KV cache compression, unbiased inner-product estimation, and compressed vector search.

The project is intentionally proposal-heavy right now. The current work is to identify the most promising algorithmic and systems directions before implementing and benchmarking them.

## Current Stage

OrbitQuant is currently in the **design and experiment-planning phase**.

What exists today:

- Research papers that define the baseline techniques.
- A detailed proposal backlog for further optimization.
- A prioritized set of ideas to implement and test.
- A roadmap for turning those proposals into experiments.

What does not exist yet:

- Production implementation code.
- CUDA kernels.
- Benchmark scripts.
- Reproducible experiment results.
- Validated performance claims for the new proposals.

Those pieces are planned next. The repository should be read as an experimental optimization lab, not as a finished quantization library.

## Motivation

Long-context LLM inference is heavily constrained by KV cache memory and bandwidth. Compressing cached keys and values can reduce serving cost and improve throughput, but small quantization errors can distort attention logits, change retrieval behavior, or degrade downstream quality.

The papers in this repository provide strong starting points:

- **QJL** shows that 1-bit sign sketches can give unbiased inner-product estimates with low metadata overhead.
- **PolarQuant** shows that random preconditioning plus polar angle quantization can compress KV caches effectively.
- **TurboQuant** combines random rotation, Lloyd-Max scalar quantization, and residual QJL to approach near-optimal distortion rates.

OrbitQuant asks the next question:

> Which further optimizations can make these methods faster, more accurate, easier to deploy, or better aligned with real LLM inference?

## Repository Contents

| File | Role |
| --- | --- |
| [`proposals.md`](./proposals.md) | Main proposal backlog for future OrbitQuant experiments. |
| [`turboquant-paper.pdf`](./turboquant-paper.pdf) | Baseline TurboQuant paper and primary target for further optimization. |
| [`quantized-johnson-lindenstrauss.pdf`](./quantized-johnson-lindenstrauss.pdf) | QJL paper, used especially for residual and unbiased inner-product ideas. |
| [`polar-quant.pdf`](./polar-quant.pdf) | PolarQuant paper, used for polar-coordinate and random-preconditioning hybrid ideas. |

## Main Research Baselines

### TurboQuant

TurboQuant rotates vectors into a distribution where scalar Lloyd-Max quantization becomes near-optimal. For unbiased inner-product estimation, it adds a QJL sketch of the residual after MSE quantization.

OrbitQuant treats TurboQuant as the main baseline to extend.

### QJL

QJL stores sign bits of a projected key while keeping the query projection unquantized. This asymmetric setup gives an unbiased estimator for query-key inner products.

OrbitQuant uses QJL as a building block for residual sketches, inner-product correction, and variable-rate scoring ideas.

### PolarQuant

PolarQuant transforms preconditioned vectors into recursive polar coordinates and quantizes the resulting angles. The angle distributions become analytically structured and increasingly concentrated at higher levels.

OrbitQuant uses PolarQuant as a source of hybrid ideas for alternative coordinate systems and angle-based residual coding.

## Proposal Themes

The proposal backlog is grouped around several optimization directions:

| Theme | Goal |
| --- | --- |
| Fast transforms | Replace dense random rotations with cheaper structured transforms. |
| Better quantizers | Reduce distortion beyond independent scalar quantization. |
| Adaptive precision | Spend bits where layers, heads, tokens, or coordinates need them most. |
| Inner-product correction | Preserve attention logits and retrieval scores more faithfully. |
| KV-cache systems work | Make compressed attention practical through layout and kernel design. |
| Cross-paper hybrids | Combine TurboQuant, QJL, and PolarQuant ideas. |
| Model-aware objectives | Optimize for token prediction quality rather than only MSE. |

The proposals are not assumed to be correct. Each one needs implementation, measurement, and rejection or refinement based on evidence.

## Near-Term Experiment Candidates

The first proposals worth implementing are the ones with high expected payoff and relatively low implementation ambiguity:

1. **Structured fast rotations**: test Hadamard/sign-flip transforms against dense random rotations.
2. **Analytical debiasing for `TurboQuant_mse`**: reduce low-bit inner-product bias without always using residual QJL.
3. **Variable-size residual QJL**: tune residual sketch dimension instead of fixing it to one bit per coordinate.
4. **Exact finite-dimension codebooks**: build Lloyd-Max codebooks for practical head dimensions such as 64 and 128.
5. **Quantized norm and radius storage**: reduce side-information overhead from norms, radii, and residual norms.
6. **Separate key/value objectives**: optimize keys for logits and values for attention-weighted reconstruction.
7. **Fused quantized attention path**: avoid materializing dequantized KV tensors during attention.

## Planned Implementation Path

### Phase 1: Reference Implementations

- Implement CPU/PyTorch reference versions of TurboQuant MSE, TurboQuant product, QJL, and selected proposal variants.
- Add deterministic unit tests for codebooks, reconstruction error, estimator bias, and variance.
- Keep implementations simple enough to inspect and compare.

### Phase 2: Synthetic Evaluation

- Test on controlled random vectors, sphere-distributed vectors, and embedding-like distributions.
- Compare measured MSE and inner-product error against theoretical expectations.
- Reject proposals that fail on simple controlled settings.

### Phase 3: KV Cache Integration

- Add integration with a small decoder-only model.
- Quantize keys and values during prefill and decode.
- Measure attention-logit error, output hidden-state error, and generated-token quality.

### Phase 4: Long-Context Benchmarks

- Evaluate on retrieval-style long-context tasks.
- Compare memory ratio, latency, and quality against baseline quantization methods.
- Track both average quality and worst-case degradation.

### Phase 5: Systems Optimization

- Add packed code layouts.
- Implement fused attention/scoring kernels.
- Benchmark real serving tradeoffs: throughput, latency, memory bandwidth, and quality.

## Success Criteria

A proposal is considered promising only if it improves at least one of the following without unacceptable regressions:

- Lower reconstruction error at the same bit budget.
- Lower inner-product error or bias at the same bit budget.
- Better long-context task quality at the same memory ratio.
- Lower runtime overhead during prefill or decode.
- Lower side-information overhead.
- Simpler deployment in existing inference stacks.
- Better vector-search recall or scoring throughput.

## Non-Goals For Now

OrbitQuant is not currently trying to be:

- A polished Python package.
- A production inference engine.
- A full replacement for FlashAttention, vLLM, or other serving stacks.
- A universal benchmark suite.
- A final paper artifact.

The immediate goal is to move from proposal backlog to measured experiments.

## Suggested Repository Descriptions

- Experimental optimization project for TurboQuant-style KV cache compression.
- Proposals and experiments for faster, sharper TurboQuant vector quantization.
- Exploring next-step optimizations for TurboQuant, QJL, and PolarQuant.
- An experimental lab for LLM KV cache quantization and compressed attention.
- Optimization backlog for near-optimal vector quantization in long-context LLMs.
- Future implementation workspace for TurboQuant extensions and benchmarks.
- Experimental roadmap for unbiased inner-product quantization and KV cache compression.
- Testing ground for structured rotations, residual sketches, and fused quantized attention.
- Research-to-implementation workspace for advanced KV cache compression ideas.
- Proposal-driven project for improving TurboQuant quality, speed, and deployability.

## License

No license file is currently included. Add one before publishing reusable implementation code.
