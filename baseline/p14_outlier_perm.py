"""P1.4 -- Outlier-aware channel permutation before the Hadamard rotation.

Quantization error concentrates on a few large ("outlier") channels. The FWHT
butterfly mixes coordinates in stages: at stage s it combines contiguous
groups of size 2^s. If the outlier channels sit inside one butterfly group,
their energy stays concentrated until late stages and produces a few very
large rotated coordinates that dominate the codebook error.

P1.4 builds a channel permutation from a calibration pass on synthetic
vectors with a few outlier channels:

1. calibration: rank channels by mean |x_i| over synthetic unit vectors whose
   (fixed) outlier channels are amplified by ``out_gain``;
2. spread: place the top-``n_out`` channels at the bit-reversed positions
   br(0), ..., br(n_out-1) (``br`` = m-bit reversal, m = log2 d). With
   k = ceil(log2 n_out) this puts exactly one outlier per butterfly group of
   size d/2^k; more generally every butterfly group of size 2^(m-k+j)
   contains at most 2^j outliers at every stage;
3. fill: the remaining channels fill the leftover positions in decreasing
   score order.

The outlier-aware permutation is fixed after calibration; only the sign
flips of the Hadamard+sign-flip rotation are re-drawn per trial.

Each rotation is a ``Rotation`` object with forward/inverse
(T = (D H ... D H) P, the FWHT sign-flip rotation applied after permuting
channels). Pure NumPy. No SciPy, no model, no GPU.
"""
from __future__ import annotations

import numpy as np

from . import rotations as rot

__all__ = [
    "calibration_vectors",
    "channel_scores",
    "outlier_aware_permutation",
    "outlier_vector",
    "permuted_hadamard_rotation",
]


def _bit_reverse(v: int, m: int) -> int:
    """Reverse the m-bit binary representation of v."""
    r = 0
    for _ in range(m):
        r = (r << 1) | (v & 1)
        v >>= 1
    return r


def calibration_vectors(
    d: int,
    n_cal: int,
    channels: np.ndarray,
    out_gain: float = 4.0,
    seed: int = 0,
) -> np.ndarray:
    """(n_cal, d) synthetic unit vectors with a few outlier channels.

    Each row is standard-normal background with the fixed ``channels``
    amplified by ``out_gain``, then normalized to unit norm. The outlier set
    is the same in every calibration vector (as in LLM activations, where
    outlier channels are consistent across tokens).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_cal, d))
    X[:, channels] *= out_gain
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def channel_scores(X: np.ndarray) -> np.ndarray:
    """Per-channel outlier score: mean |x_i| over the calibration set."""
    return np.mean(np.abs(X), axis=0)


def outlier_aware_permutation(
    d: int,
    scores: np.ndarray,
    n_out: int,
) -> np.ndarray:
    """Permutation spreading the top-n_out channels across FWHT butterfly groups.

    Returns ``perm`` with ``perm[p]`` = source channel placed at output
    position ``p`` (``x[perm]`` builds the permuted vector directly). The top
    ``n_out`` channels by ``scores`` (highest magnitude) are placed at the
    bit-reversed positions ``br(0..n_out-1)``; the rest fill the remaining
    positions in decreasing score order.

    Spreading guarantee (m = log2 d, k = ceil(log2 n_out), d a power of two):
    every FWHT butterfly group of size 2^(m-k+j) contains at most 2^j outlier
    channels, for all j >= 0 -- in particular one per group of size d/2^k.
    """
    if d & (d - 1):
        raise ValueError(f"d={d} must be a power of two")
    if n_out < 1:
        return np.arange(d)
    if n_out > d:
        raise ValueError(f"n_out={n_out} must be <= d={d}")
    m = int(np.log2(d))
    k = min(int(np.ceil(np.log2(n_out))), m)
    block = d >> k  # butterfly group size with at most one outlier
    s = m - k  # log2(block); offset lives inside the block
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    pos = np.full(d, -1, dtype=np.int64)
    for i in range(n_out):
        blk = i & ((1 << k) - 1)
        off = _bit_reverse(blk & (block - 1), s)
        pos[blk * block + off] = int(order[i])
    fill = np.nonzero(pos < 0)[0]
    pos[fill] = order[n_out:]  # remaining channels, decreasing score
    return pos


def outlier_vector(
    d: int,
    channels: np.ndarray,
    out_gain: float = 4.0,
    seed: int = 0,
) -> np.ndarray:
    """Fixed adversarial unit vector with the outlier channels amplified.

    Deterministic given seed; used as the fixed x in the rotation checks.
    Clustered outlier channels (at the head of the vector) are the worst case
    for a fixed FWHT without a permutation.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d)
    v[channels] *= out_gain
    return v / np.linalg.norm(v)


def permuted_hadamard_rotation(
    d: int,
    rng: np.random.Generator,
    perm: np.ndarray | None = None,
    rounds: int = 1,
) -> rot.Rotation:
    """Hadamard+sign-flip rotation with an optional pre-permutation.

    T = (D_k H ... D_1 H) P: the FWHT sign-flip rotation of ``rotations``
    applied after permuting channels by ``perm`` (``perm[p]`` = source
    channel at output position ``p``, so ``x[perm]`` is the permuted vector;
    inverse applies ``y[argsort(perm)]``). With ``perm=None`` this is exactly
    ``rot.hadamard_sign_flip`` (no permutation). Returns a ``Rotation``
    object with forward/inverse.
    """
    hd = rot.hadamard_sign_flip(d, rng, rounds=rounds)
    if perm is None:
        return hd
    p = np.asarray(perm, dtype=np.int64)
    inv = np.argsort(p)
    return rot.Rotation(
        lambda x: hd.forward(np.asarray(x, dtype=np.float64)[p]),
        lambda x: hd.inverse(np.asarray(x, dtype=np.float64))[inv],
        "hd+perm",
    )
