"""Scalar codebooks for TurboQuant's rotated-coordinate distribution.

Two paths, side by side:

1. ``beta_lloyd_max`` -- the exact finite-dimension Lloyd-Max codebook for the
   true coordinate distribution (TurboQuant Lemma 1):
       f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
   This is what proposal P0.3 wants, and it is the theoretically correct
   codebook for Haar random rotations.

2. ``gaussian_closed_form`` -- the high-dimension Gaussian asymptotic codebook
   whose centroids are given in closed form in the paper (Sec 3.1):
       b=1:  +/- sqrt(2/pi)/sqrt(d)
       b=2:  {+/- 0.453/sqrt(d), +/- 1.51/sqrt(d)}
   These match the paper's published distortion figures (0.36, 0.117, ...) but
   are only exact as d -> infinity.

The two are compared in the baseline to quantify how much the asymptotic
shortcut costs at practical head dimensions (d = 64, 128).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "beta_pdf",
    "beta_lloyd_max",
    "gaussian_closed_form",
    "quantize",
    "dequantize",
]


# --------------------------------------------------------------------------- #
# Coordinate distribution (TurboQuant Lemma 1)
# --------------------------------------------------------------------------- #
def beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """Evaluate the exact coordinate density f_X(x) for dimension d.

    f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
    for x in [-1, 1]. Computed in log space for numerical stability.
    """
    x = np.asarray(x, dtype=np.float64)
    # log of the normalization constant
    log_norm = (
        np.log(np.sqrt(np.pi))
        + np.log(math.gamma((d - 1) / 2))
        - np.log(math.gamma(d / 2))
    )
    # (1 - x^2)^((d-3)/2), clipped to avoid nan at |x| -> 1
    inside = np.clip(1.0 - x * x, 1e-16, None)
    log_pdf = -log_norm + ((d - 3) / 2) * np.log(inside)
    return np.exp(log_pdf)


# --------------------------------------------------------------------------- #
# Lloyd-Max (continuous 1-D k-means) for a symmetric density on [-1, 1]
# --------------------------------------------------------------------------- #
def _lloyd_max_symmetric(
    density: callable,
    n_centroids: int,
    grid: int = 200_001,
    tol: float = 1e-12,
    max_iter: int = 500,
) -> np.ndarray:
    """Solve the 1-D k-means problem for a symmetric density on [-1, 1].

    Returns the ``n_centroids`` positive centroids (ascending), which are
    mirrored to form the full symmetric codebook. Uses the Lloyd-Max fixed
    point: cell boundaries are midpoints between adjacent centroids, and each
    centroid is the conditional mean of the density within its cell.

    Robustness: initialized from equal-probability-mass quantile cells (which
    are non-empty for a log-concave density), then Lloyd iterations. Enforces
    sorted, non-empty cells and convergence; raises if the fixed point is not
    reached or the codebook is degenerate.
    """
    t = np.linspace(0.0, 1.0, grid)
    w = density(t)  # unnormalized density on [0,1]
    # raw cumulative mass / first moment over [0,1] (trapezoid integration)
    mass = np.cumsum(w) - 0.5 * w - 0.5 * w[0]
    first = np.cumsum(t * w) - 0.5 * (t * w) - 0.5 * (t[0] * w[0])
    M = mass[-1]

    def _idx(v: float) -> int:
        return min(int(np.searchsorted(t, v)), grid - 1)

    k = n_centroids
    # init: split the probability mass into k equal-mass cells (quantiles).
    # Guarantees non-empty cells, unlike equal-coordinate spacing which leaves
    # zero-mass cells near the origin for sharply peaked densities.
    edges = np.empty(k + 1)
    edges[0] = 0.0
    for j in range(1, k):
        edges[j] = t[np.searchsorted(mass, j / k * M)]
    edges[k] = 1.0
    c = np.empty(k)
    for i in range(k):
        lo, hi = _idx(edges[i]), _idx(edges[i + 1])
        dm = mass[hi] - mass[lo]
        c[i] = (first[hi] - first[lo]) / dm

    converged = False
    for _ in range(max_iter):
        # boundaries between adjacent centroids (midpoints)
        b = np.empty(k + 1)
        b[0] = 0.0
        b[1:-1] = (c[:-1] + c[1:]) / 2.0
        b[-1] = 1.0
        # new centroids = conditional means; re-seed any empty cell from the
        # equal-mass quantile grid to keep the partition non-degenerate
        c_new = np.empty(k)
        for i in range(k):
            lo, hi = _idx(b[i]), _idx(b[i + 1])
            dm = mass[hi] - mass[lo]
            if dm <= 0:
                # empty cell: split the largest-mass cell by re-seeding this
                # centroid at the midpoint of the largest cell
                sizes = np.array([mass[_idx(b[j + 1])] - mass[_idx(b[j])]
                                  for j in range(k)])
                jmax = int(np.argmax(sizes))
                c_new[i] = (b[jmax] + b[jmax + 1]) / 2.0
            else:
                c_new[i] = (first[hi] - first[lo]) / dm
        if np.max(np.abs(c_new - c)) < tol:
            c = c_new
            converged = True
            break
        c = c_new

    if not converged:
        raise RuntimeError(
            f"Lloyd-Max did not converge in {max_iter} iterations (k={k})"
        )
    if not np.all(np.diff(c) > 0):
        raise RuntimeError(f"Lloyd-Max produced non-sorted codebook: {c}")
    return c


def beta_lloyd_max(d: int, b: int, grid: int = 100_001, max_iter: int = 500) -> np.ndarray:
    """Exact finite-d Beta Lloyd-Max codebook for bit-width b.

    Returns 2**b centroids, symmetric about zero, sorted ascending.
    ``max_iter`` is passed through to the Lloyd-Max fixed-point solver
    (default 500; larger values help wide codebooks, e.g. b >= 5).
    """
    n_pos = 2 ** (b - 1)  # positive-side centroids
    pos = _lloyd_max_symmetric(lambda x: beta_pdf(x, d), n_pos, grid=grid,
                               max_iter=max_iter)
    neg = -pos
    return np.concatenate([neg[::-1], pos])


# --------------------------------------------------------------------------- #
# Gaussian closed-form codebook (paper Sec 3.1)
# --------------------------------------------------------------------------- #
def gaussian_closed_form(d: int, b: int) -> np.ndarray:
    """Closed-form Gaussian-asymptotic codebook for bit-width b.

    Matches the paper's published distortion figures at b = 1, 2. Only exact
    in the limit d -> infinity.
    """
    if b == 1:
        c = np.array([np.sqrt(2.0 / np.pi)])
    elif b == 2:
        c = np.array([0.453, 1.51])
    else:
        raise ValueError(f"closed-form centroids only defined for b=1,2 (got b={b})")
    c = c / np.sqrt(d)
    return np.concatenate([-c[::-1], c])


# --------------------------------------------------------------------------- #
# Quantization primitives (operate on normalized, unit-norm vectors)
# --------------------------------------------------------------------------- #
def quantize(x: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Quantize each coordinate to the nearest centroid index (b-bit ints)."""
    # x: (..., d); codebook: (2**b,) sorted ascending
    idx = np.argmin(np.abs(x[..., np.newaxis] - codebook), axis=-1)
    return idx


def dequantize(idx: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Reconstruct coordinates from centroid indices."""
    return codebook[idx]
