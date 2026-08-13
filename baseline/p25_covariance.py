"""P2.5 -- Covariance-aware rotations.

A rotation derived from an empirical covariance target

    T = D2 U H D1 P

where U is the eigenbasis of a covariance matrix (columns sorted by
descending eigenvalue), H is the normalized Hadamard transform (FWHT
butterfly), and P is a *balancing permutation* that interleaves
high-variance and low-variance coordinates so that every FWHT butterfly
pair mixes one large and one small eigenvalue. D1, D2 are random sign
flips and P is re-shuffled within the two variance bands per trial, so
the rotation is a randomized family (like the randomized Hadamard).

For signals whose covariance matches the calibration target, T
decorrelates the coordinates (U) and balances their energy (H, P): the
rotated coordinates are approximately isotropic with variance
trace(Sigma)/d, which keeps scalar quantization errors balanced. Unlike
Haar, the transform is *adaptive*: it is not a universal mixer, so the
standard Beta-coordinate checks on an adversarial fixed vector degrade.

The covariance is *empirical*: it is estimated from n_cal synthetic
calibration samples of a generative model with k_out large-magnitude
outlier channels (eigenvalue ratio out_ratio).

All rotations are ``Rotation`` objects (forward/inverse). Pure NumPy;
no SciPy, no model, no GPU.
"""
from __future__ import annotations

import numpy as np

from . import rotations as rot
from .protocol import random_rotation

__all__ = [
    "synthetic_covariance_data",
    "empirical_eigenbasis",
    "balancing_permutation",
    "covariance_rotation",
    "top_k_variance_fraction",
]


def synthetic_covariance_data(
    d: int,
    rng: np.random.Generator,
    n_cal: int = 2048,
    k_out: int = 4,
    out_ratio: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic calibration data with a few large-magnitude outlier channels.

    Generative model: rows X = (L^{1/2} Z) Q^T with Z ~ N(0, I_{n_cal x d}),
    Q a fixed random orthogonal 'channel' basis, and L diagonal with the
    first k_out eigenvalues equal to out_ratio and the rest 1. Returns

        (X, Sigma, Q, lam)

    where X is the (n_cal, d) calibration data and Sigma = X^T X / n_cal is
    the *empirical* covariance the rotation is derived from; Q, lam are the
    ground-truth model parameters (used to draw fresh evaluation samples).
    """
    Q = random_rotation(d, rng)
    lam = np.ones(d)
    lam[:k_out] = out_ratio
    Z = rng.standard_normal((n_cal, d))
    X = (Z * np.sqrt(lam)) @ Q.T
    Sigma = X.T @ X / n_cal
    return X, Sigma, Q, lam


def empirical_eigenbasis(Sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and eigenbasis U of a covariance matrix."""
    w, U = np.linalg.eigh(Sigma)
    order = np.argsort(w)[::-1]
    return w[order], U[:, order]


def balancing_permutation(
    eigenvalues: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Balancing permutation pi with (P x)_k = x[pi[k]].

    Coordinates are sorted by descending eigenvalue; the top and bottom
    halves are shuffled independently and interleaved, so every FWHT
    butterfly pair (2j, 2j+1) mixes one high-variance and one low-variance
    coordinate. Requires an even d.
    """
    d = eigenvalues.shape[0]
    if d % 2 != 0:
        raise ValueError(f"d={d} must be even for the balancing permutation")
    order = np.argsort(eigenvalues)[::-1]
    half = d // 2
    hi = rng.permutation(order[:half])
    lo = rng.permutation(order[half:])
    pi = np.empty(d, dtype=np.int64)
    pi[0::2] = hi
    pi[1::2] = lo
    return pi


def covariance_rotation(
    eigenvalues: np.ndarray,
    U: np.ndarray,
    rng: np.random.Generator,
) -> rot.Rotation:
    """One randomized covariance-aware rotation T = D2 U H D1 P.

    Eigenbasis U (columns sorted by descending eigenvalue), Hadamard H via
    the FWHT butterfly, balancing permutation P, and fresh random sign flips
    D1, D2 plus a fresh balancing permutation per call, so repeated calls
    give the randomized rotation family used for the benchmark.

    O(d^2) forward/inverse (the eigenbasis U is dense; it is the learned
    object). Requires d a power of two for the FWHT.
    """
    d = U.shape[0]
    if d & (d - 1):
        raise ValueError(f"d={d} must be a power of two for the FWHT")
    pi = balancing_permutation(eigenvalues, rng)
    inv_pi = np.argsort(pi)
    s1 = rng.choice([-1.0, 1.0], size=d)
    s2 = rng.choice([-1.0, 1.0], size=d)

    def _h(x: np.ndarray) -> np.ndarray:
        return rot.fwht(x) / np.sqrt(d)

    def forward(x):
        y = x[pi]              # P  (balancing permutation)
        y = s1 * y             # D1 (random signs)
        y = _h(y)              # H  (normalized Hadamard)
        y = U @ y              # U  (empirical eigenbasis)
        return s2 * y          # D2 (random signs)

    def inverse(x):
        y = s2 * x             # D2 (symmetric)
        y = U.T @ y            # U^T
        y = _h(y)              # H^T = H
        y = s1 * y             # D1
        return y[inv_pi]       # P^T

    return rot.Rotation(forward, inverse, "covUHP")


def top_k_variance_fraction(eigenvalues: np.ndarray, k: int) -> float:
    """Fraction of total variance carried by the top-k eigen-directions."""
    return float(np.sum(eigenvalues[:k]) / np.sum(eigenvalues))
