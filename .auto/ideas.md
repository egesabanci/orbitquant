# Autoresearch ideas backlog

_Deferred-but-promising ideas and cross-cutting observations. Prune tried
entries as experiments land._

## Open / promising
- **Noise-limited estimator design (from exp 11)**: softmax attention is
  dominated by per-key logit-noise VARIANCE. Probe the attention-KL-per-byte
  tradeoff of variance-reducing codes (residual/QJL at V+1 bits, fine bit
  splits) against the raw b=2 design on `attention_metrics()` — NOT global
  score scaling (amplifies noise by 1/c^2).
- **Real-KV deployment (GPU box)**: per-layer calibrated norm ranges (P1.3/
  P2.5) to avoid range-clipping at wide dynamic ranges; empirical codebooks
  re-checked on real non-Beta marginals; P0.4 harness to validate defaults
  (6-bit norm, raw estimator, hdhdh rotation) AND to confirm `tq_attn_kl_b2`
  predicts real-model softmax-KL; P2.1 fused kernel.
- **Settled in this regime (do NOT re-attempt)**: Lloyd-on-log-norm norm
  codebook (+0.16% worse); empirical direction codebooks for hdhdh (~0%);
  norm-preserving reconstruction (worse MSE); any norm-range search beyond
  ±1.2; further FWHT micro-optimization (at numpy-call floor); multiplicative
  debias as an attention lever (refuted exp 11 — amplifies per-key noise).

## Established / do NOT re-attempt
- P2.8 OCTOPUS triplets lose to scalar at every matched budget (near-independent
  Beta coords).
- P2.9 joint rounding: ~0-6%, only patches a worse codec.
- Single-round Hadamard / permutation rotations fail distribution gates.
- Pure-Python FWHT slow (that IS the open compute target above).
- Real-KV / covariance rotations / GPU kernels: out of scope (needs GPU box).
