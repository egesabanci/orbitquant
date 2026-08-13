"""Shared statistical benchmark protocol for OrbitQuant proposals.

Centralizes the protocol used by the baseline so every proposal is measured
with the exact same methodology:

- Fixed/adversarial input vector x (worst-case), averaged over independent
  sign-corrected Haar-QR random rotations (matches the paper's claim: fixed x,
  random Pi -> Pi*x has Beta coordinates).
- Fixed x,y with nonzero dot and independent sketches for QJL/bias checks.

Pure NumPy. No SciPy, no model, no GPU.
"""
from __future__ import annotations

import numpy as np

from . import codebooks as cb

__all__ = [
    "random_rotation",
    "fixed_vector",
    "beta_cdf",
    "beta_ks",
    "sample_beta",
    "mse_after_quantize",
    "inner_product_bias_ratio",
]


# --------------------------------------------------------------------------- #
# Rotation protocol
# --------------------------------------------------------------------------- #
def random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """Independent Haar random rotation (sign-corrected QR of a Gaussian).

    QR of a Gaussian matrix is Haar up to column signs; fixing the sign of each
    column by the sign of the corresponding R diagonal yields the unique QR and
    a Haar-distributed orthogonal matrix (diagonal treated as +1 when zero).
    """
    a = rng.standard_normal((d, d))
    q, r = np.linalg.qr(a)
    q *= np.where(np.diag(r) >= 0, 1.0, -1.0)
    return q


def fixed_vector(d: int, kind: str = "e1") -> np.ndarray:
    """A fixed, adversarial unit vector (worst-case input)."""
    x = np.zeros(d)
    if kind == "e1":
        x[0] = 1.0
    elif kind == "all_equal":
        x[:] = 1.0 / np.sqrt(d)
    else:
        raise ValueError(kind)
    return x


# --------------------------------------------------------------------------- #
# Beta distribution (pure NumPy)
# --------------------------------------------------------------------------- #
def beta_cdf(d: int, samples: np.ndarray, grid: int = 200_001) -> np.ndarray:
    """Exact CDF of the coordinate law (TurboQuant Lemma 1) by grid integration.

    Integrates the closed-form beta_pdf on a fine grid and interpolates at the
    sample points (matches the exact Beta CDF to ~6e-10).
    """
    t = np.linspace(-1.0, 1.0, grid)
    w = cb.beta_pdf(t, d)
    cdf = np.cumsum(w) - 0.5 * w - 0.5 * w[0]
    cdf = cdf / cdf[-1]
    return np.interp(samples, t, cdf)


def beta_ks(d: int, samples: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic vs the exact Beta cdf."""
    x = np.sort(samples)
    cdf = beta_cdf(d, x)
    n = len(x)
    ecdf = np.arange(1, n + 1) / n
    return float(np.max(np.abs(ecdf - cdf)))


def sample_beta(a: float, b: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from Beta(a,b) via the gamma-ratio identity (pure NumPy)."""
    g1 = rng.standard_gamma(a, size=size)
    g2 = rng.standard_gamma(b, size=size)
    return g1 / (g1 + g2)


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def mse_after_quantize(
    x: np.ndarray,
    codebook: np.ndarray,
    rotation: np.ndarray | None = None,
) -> float:
    """Per-vector MSE after quantizing x (optionally rotated) with a codebook.

    x: unit vector (d,). If rotation given, quantize in the rotated basis and
    rotate back; else quantize x directly.
    """
    y = rotation @ x if rotation is not None else x
    idx = cb.quantize(y, codebook)
    yhat = cb.dequantize(idx, codebook)
    if rotation is not None:
        yhat = rotation.T @ yhat
    return float(np.sum((x - yhat) ** 2))


def inner_product_bias_ratio(
    x: np.ndarray,
    y: np.ndarray,
    codebook: np.ndarray,
    n_rot: int,
    rng: np.random.Generator,
) -> float:
    """Regression-slope bias ratio E[<y,Q(x)>]/<y,x> over independent rotations."""
    true = float(np.dot(y, x))
    d = x.shape[0]
    ests = np.empty(n_rot)
    for i in range(n_rot):
        P = random_rotation(d, rng)
        yx = P @ x
        idx = cb.quantize(yx, codebook)
        xhat = P.T @ cb.dequantize(idx, codebook)
        ests[i] = np.dot(y, xhat)
    return float(np.mean(ests) / true)
