"""P2.9 — Joint rounding for small-block codecs.

Small-block codec under test (spherical block codec):

    y_block in R^B  ->  encode (n = ||y_block||, d = y_block / n)

- The B direction coordinates are scalar-quantized with the exact finite-d
  Beta Lloyd-Max codebook of dimension B (``codebooks.beta_lloyd_max(B, b)``):
  for an isotropic d-vector rotated by a Haar rotation, each coordinate of a
  B-block direction is distributed exactly like one coordinate of a uniform
  unit vector on S^(B-1), i.e. the Beta((B-1)/2, (B-1)/2) law.
- The block norm is scalar-quantized with a 1-D Lloyd-Max codebook fitted
  (pure NumPy fixed point) to the exact norm law n^2 ~ Beta(B/2, (d-B)/2).

    decode: block_hat = n_hat * normalize(Q(d))

The direction reconstruction *renormalizes* the dequantized coordinates onto
the sphere, which couples the B scalar choices: the tuple of per-coordinate
nearest centroids does not necessarily minimize the reconstructed block error
(block error = n^2 * chordal error + constant norm term). ``encode_joint``
therefore enumerates a tiny local candidate set around the nearest centroids
(index +/- k per coordinate, clipped to the codebook) and picks the tuple with
the smallest reconstructed block error; ``encode_independent`` is the plain
per-coordinate nearest-centroid baseline.

Protocol: fixed adversarial x = e1, averaged over independent sign-corrected
Haar-QR rotations per trial. Pure NumPy, no SciPy.
"""
from __future__ import annotations

import numpy as np

from . import codebooks as cb
from . import protocol as pr

__all__ = [
    "lloyd_max_positive",
    "spherical_block_codebooks",
    "encode_independent",
    "encode_joint",
    "block_codec_mse",
]

_NORM_FLOOR = 1e-12


# --------------------------------------------------------------------------- #
# 1-D Lloyd-Max on a calibration sample (pure NumPy)
# --------------------------------------------------------------------------- #
def lloyd_max_positive(
    samples: np.ndarray,
    n_centroids: int,
    max_iter: int = 300,
    tol: float = 1e-13,
) -> np.ndarray:
    """1-D Lloyd-Max (k-means) centroids for a non-negative-valued density.

    Fixed-point iteration on a large calibration sample: assign every sample to
    its nearest centroid, then set each centroid to the mean of its assigned
    samples. Returns ``n_centroids`` ascending centroids.
    """
    s = np.sort(np.asarray(samples, dtype=np.float64).ravel())
    n = s.shape[0]
    if n_centroids <= 1:
        return np.array([np.mean(s)])
    if s[-1] - s[0] <= 0.0:
        return np.full(n_centroids, s[0])
    edges = np.linspace(0.0, n, n_centroids + 1).astype(int)
    c = np.array([np.mean(s[edges[j]:edges[j + 1]]) for j in range(n_centroids)])
    for _ in range(max_iter):
        idx = np.argmin(np.abs(s[:, None] - c[None, :]), axis=1)
        c_new = np.empty_like(c)
        for j in range(n_centroids):
            m = idx == j
            c_new[j] = np.mean(s[m]) if np.any(m) else c[j]
        if np.max(np.abs(c_new - c)) < tol:
            c = c_new
            break
        c = c_new
    return c


def symmetric_lloyd_max(
    samples: np.ndarray,
    bits: int,
    cal_n: int = 200_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Symmetric Lloyd-Max codebook (2**bits centroids) from exact-law samples.

    Sample-based alternative to ``codebooks.beta_lloyd_max`` for laws whose
    density is unbounded at +-1 (e.g. the arcsine law of a block=2 direction
    coordinate), where the grid-based solver does not converge. Fits the
    positive side on folded |x| samples and mirrors.
    """
    pos = lloyd_max_positive(np.abs(samples), 2 ** (bits - 1))
    return np.concatenate([-pos[::-1], pos])


# --------------------------------------------------------------------------- #
# Spherical block codec
# --------------------------------------------------------------------------- #
def spherical_block_codebooks(
    d: int,
    block: int,
    bits_dir: int,
    bits_norm: int,
    rng: np.random.Generator,
    cal_n: int = 200_000,
) -> dict:
    """Codebook pair for the spherical block codec.

    Direction codebook: exact finite-d Beta Lloyd-Max for one coordinate of a
    uniform unit vector in R^block (matches the protocol's Beta-coordinate
    law), except block=2 where the arcsine law's singularity at +-1 prevents
    the grid-based solver from converging -- there the codebook is fitted by
    sample-based Lloyd-Max on the exact law (x = 2*Beta(1/2,1/2) - 1). Norm
    codebook: Lloyd-Max on samples from the exact block-norm law
    n ~ sqrt(Beta(block/2, (d - block)/2)).
    """
    if block == 2:
        b_samp = pr.sample_beta(0.5, 0.5, cal_n, rng)
        dir_cbk = symmetric_lloyd_max(2.0 * b_samp - 1.0, bits_dir)
    else:
        dir_cbk = cb.beta_lloyd_max(block, bits_dir)
    a, b = block / 2.0, (d - block) / 2.0
    n_samp = np.sqrt(pr.sample_beta(a, b, cal_n, rng))
    norm_cbk = lloyd_max_positive(n_samp, 2 ** bits_norm)
    return {
        "dir": dir_cbk,
        "norm": norm_cbk,
        "bits_dir": bits_dir,
        "bits_norm": bits_norm,
        "block": block,
    }


def encode_independent(Y: np.ndarray, codebooks: dict) -> np.ndarray:
    """Independent rounding: per-coordinate nearest centroid, then renormalize.

    Y: (n_blocks, block) rotated coordinates. Returns reconstructed blocks
    (n_blocks, block).
    """
    d_cbk = codebooks["dir"]
    n_cbk = codebooks["norm"]
    norm = np.maximum(np.linalg.norm(Y, axis=1), _NORM_FLOOR)
    D = Y / norm[:, None]
    Dhat = cb.dequantize(cb.quantize(D, d_cbk), d_cbk)
    Dhat /= np.maximum(np.linalg.norm(Dhat, axis=1, keepdims=True), _NORM_FLOOR)
    nhat = n_cbk[np.argmin(np.abs(norm[:, None] - n_cbk[None, :]), axis=1)]
    return nhat[:, None] * Dhat


def encode_joint(Y: np.ndarray, codebooks: dict, k: int) -> np.ndarray:
    """Joint rounding: tiny local candidate enumeration around nearest centroids.

    For each block, the per-coordinate nearest centroid index is computed, then
    a local index set (i +/- k, clipped to the codebook) is formed per
    coordinate and the Cartesian product of tuples is enumerated. The tuple
    minimizing the reconstructed block error is chosen. The norm term is
    constant across tuples, so minimizing the chordal error ||D - Dhat||^2 is
    equivalent to minimizing the block error.
    """
    d_cbk = codebooks["dir"]
    n_cbk = codebooks["norm"]
    L = d_cbk.shape[0]
    B = Y.shape[1]
    norm = np.maximum(np.linalg.norm(Y, axis=1), _NORM_FLOOR)
    D = Y / norm[:, None]
    i0 = cb.quantize(D, d_cbk)  # (n_blocks, B) nearest indices
    offs = np.stack(
        np.meshgrid(*([np.arange(-k, k + 1)] * B), indexing="ij"), axis=-1
    ).reshape(-1, B)  # (C, B) offset tuples
    cand = np.clip(i0[:, None, :] + offs[None, :, :], 0, L - 1)
    Dc = d_cbk[cand]  # (n_blocks, C, B) dequantized candidates
    Dc /= np.maximum(np.linalg.norm(Dc, axis=-1, keepdims=True), _NORM_FLOOR)
    err = np.sum((Dc - D[:, None, :]) ** 2, axis=-1)  # (n_blocks, C)
    best = cand[np.arange(Y.shape[0]), np.argmin(err, axis=1), :]
    Dhat = cb.dequantize(best, d_cbk)
    Dhat /= np.maximum(np.linalg.norm(Dhat, axis=1, keepdims=True), _NORM_FLOOR)
    nhat = n_cbk[np.argmin(np.abs(norm[:, None] - n_cbk[None, :]), axis=1)]
    return nhat[:, None] * Dhat


# --------------------------------------------------------------------------- #
# Protocol benchmark
# --------------------------------------------------------------------------- #
def block_codec_mse(
    d: int,
    block: int,
    codebooks: dict,
    n_rot: int,
    rng: np.random.Generator,
    ks: tuple = (1, 2),
) -> tuple:
    """Per-coordinate MSE under the shared protocol.

    Fixed adversarial x = e1; each trial uses an independent sign-corrected
    Haar-QR rotation. The rotated vector is padded with zeros to a whole number
    of blocks; MSE is computed over the original d coordinates only.

    Returns (mse_independent, {k: mse_joint}) with mse_joint for each k in ks.
    """
    x = pr.fixed_vector(d)
    pad = (-d) % block
    n_blocks = (d + pad) // block
    mse_i = 0.0
    mse_j = {k: 0.0 for k in ks}
    for _ in range(n_rot):
        y = pr.random_rotation(d, rng) @ x
        Y = np.pad(y, (0, pad)).reshape(n_blocks, block)
        Yh_i = encode_independent(Y, codebooks)
        mse_i += float(np.sum((Y - Yh_i).ravel()[:d] ** 2))
        for k in ks:
            Yh_j = encode_joint(Y, codebooks, k)
            mse_j[k] += float(np.sum((Y - Yh_j).ravel()[:d] ** 2))
    denom = n_rot * d
    return mse_i / denom, {k: v / denom for k, v in mse_j.items()}
