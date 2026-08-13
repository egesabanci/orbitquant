"""P1 corrections: analytical MSE debiasing and norm-preserving reconstruction.

These are cheap corrections applied after dequantization that do not change the
codebook or bitstream.

- ``debiased_quantize`` (P1.7): corrects the inner-product bias of the MSE
  quantizer. At b=1 the MSE estimator shrinks inner products by ~2/pi (paper
  Sec 3.2); the correction is a multiplicative factor estimated per codebook.
- ``norm_preserving_quantize`` (P1.8): renormalizes the dequantized vector to
  the original norm before scoring, preserving the key norm that attention sees.

Pure NumPy. No model, no GPU.
"""
from __future__ import annotations

import numpy as np

from . import codebooks as cb

__all__ = [
    "bias_correction_factor",
    "debiased_quantize",
    "norm_preserving_quantize",
]


def bias_correction_factor(
    d: int, b: int, n_rot: int = 2000, seed: int = 0
) -> float:
    """Empirical inner-product bias ratio E[<y,Q(x)>]/<y,x> for the MSE codebook.

    At b=1 this is ~2/pi (paper Sec 3.2); the reciprocal is the multiplicative
    correction. Measured with the standard protocol (fixed x,y, independent
    rotations). Deterministic given seed.
    """
    from . import protocol as pr

    rng = np.random.default_rng(seed)
    x = pr.fixed_vector(d)
    y = pr.fixed_vector(d, kind="all_equal")
    cbk = cb.beta_lloyd_max(d, b)
    true = float(np.dot(y, x))
    ests = np.empty(n_rot)
    for i in range(n_rot):
        P = pr.random_rotation(d, rng)
        yx = P @ x
        idx = cb.quantize(yx, cbk)
        xhat = P.T @ cb.dequantize(idx, cbk)
        ests[i] = np.dot(y, xhat)
    return float(np.mean(ests) / true)


def debiased_quantize(
    x: np.ndarray,
    codebook: np.ndarray,
    rotation: np.ndarray | None = None,
    correction: float | None = None,
) -> np.ndarray:
    """Quantize x and correct the inner-product bias by a multiplicative factor.

    If ``correction`` is None, the reciprocal of the codebook's measured bias
    ratio is used (computed on demand). Returns the debiased reconstruction.
    """
    y = rotation @ x if rotation is not None else x
    idx = cb.quantize(y, codebook)
    yhat = cb.dequantize(idx, codebook)
    if rotation is not None:
        yhat = rotation.T @ yhat
    if correction is None:
        d = x.shape[0]
        b = int(round(np.log2(len(codebook))))
        correction = 1.0 / bias_correction_factor(d, b)
    return correction * yhat


def norm_preserving_quantize(
    x: np.ndarray,
    codebook: np.ndarray,
    rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Quantize x and renormalize the reconstruction to the original norm.

    Preserves the key norm that attention actually sees. May slightly worsen raw
    coordinate MSE while improving logits (paper P1.8 tradeoff).
    """
    y = rotation @ x if rotation is not None else x
    idx = cb.quantize(y, codebook)
    yhat = cb.dequantize(idx, codebook)
    if rotation is not None:
        yhat = rotation.T @ yhat
    norm = np.linalg.norm(x)
    yhat_norm = np.linalg.norm(yhat)
    if yhat_norm > 0:
        yhat = yhat * (norm / yhat_norm)
    return yhat
