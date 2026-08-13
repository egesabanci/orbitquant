"""P2.5 -- Covariance-aware rotations.

A *fixed* rotation calibrated offline from an empirical covariance target
(proposals.md P2.5: "derive fixed rotations ... offline"). Two forward
conventions are implemented and benchmarked, both with the same ingredients
-- eigenbasis U (``eigh`` column convention: z = U^T x are the
eigen-coordinates whose variances are the eigenvalues), balancing
permutation P (interleaves high- and low-variance eigen-coordinates so
every FWHT butterfly pair mixes one large and one small eigenvalue),
frozen sign flip D1, normalized Hadamard H (FWHT butterfly):

    covUHP  T = U H D1 P U^T    the proposal's "U*H*P" product structure:
                                decorrelate into eigen-coordinates (U^T),
                                balance (P), mix (H), map back (U). The
                                literal U-H-P composition acts on
                                eigen-coordinates, so P's high/low
                                balancing is real. The final U re-enters
                                the data basis and mixes H's off-diagonal
                                covariance into the diagonal (measured).
    covHP   T = H D1 P U^T      transpose/order convention: decorrelate,
                                balance, mix, and stay in the mixed basis
                                (no final U). The transformed-covariance
                                diagonal is then exactly trace(Cov(x))/d:
                                    diag(T Cov(x) T^T)
                                      = diag(H P U^T Cov(x) U P^T H)
                                      = (1/d) sum_i (U^T Cov(x) U)_ii
                                        + (1/d) sum_{i!=j} +- (U^T Cov(x) U)_ij
                                i.e. guaranteed balanced up to the small
                                eigen-estimation residual of U^T Cov(x) U.

U, P and D1 are drawn once from calibration and frozen; the returned
``Rotation`` is the deployed transform, evaluated on fresh data (fixed
offline rotation). Both conventions are reported in the runner, with the
order/transpose convention explained above.

The covariance is *empirical*: it is estimated from n_cal synthetic
calibration samples of a generative model with k_out large-magnitude
outlier channels whose eigenvalues decrease geometrically (largest =
out_ratio, halving per channel), so the eigenbasis is well determined.

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
    "calibrate_covariance_rotation",
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
    first k_out eigenvalues equal to out_ratio * 0.5^j (geometrically
    decreasing, well separated) and the rest 1. Returns

        (X, Sigma, Q, lam)

    where X is the (n_cal, d) calibration data and Sigma = X^T X / n_cal is
    the *empirical* covariance the rotation is derived from; Q, lam are the
    ground-truth model parameters (used to draw fresh evaluation samples).
    """
    Q = random_rotation(d, rng)
    lam = np.ones(d)
    lam[:k_out] = out_ratio * (0.5 ** np.arange(k_out))
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


def calibrate_covariance_rotation(
    eigenvalues: np.ndarray,
    U: np.ndarray,
    rng: np.random.Generator,
    convention: str = "hp",
) -> rot.Rotation:
    """Calibrate ONE fixed covariance-aware rotation (offline).

    Called once after estimating the empirical covariance; the balancing
    permutation P and the sign flip D1 are drawn from ``rng`` exactly once
    and frozen, so the returned ``Rotation`` is a fixed object -- the
    deployed transform for all subsequent evaluation (P2.5 "fixed offline
    rotations"). Call again only to re-calibrate from new data.

    ``convention`` selects the forward order (see the module docstring):

    - "hp"  (default): T = H D1 P U^T. Decorrelate (U^T), balance (P),
      mix (H); no final U, so diag(T Cov(x) T^T) = trace(Cov(x))/d exactly
      up to the eigen-estimation residual -- guaranteed balanced rotated
      coordinates.
    - "uhp": T = U H D1 P U^T. The proposal's literal "U*H*P" product
      structure: decorrelate, balance, mix, then map back through U. The
      final U re-enters the data basis; H's off-diagonal covariance then
      leaks into the diagonal (measured in the runner's balance check).

    O(d^2) forward/inverse (U is dense; it is the learned object). Requires
    d a power of two for the FWHT.
    """
    d = U.shape[0]
    if d & (d - 1):
        raise ValueError(f"d={d} must be a power of two for the FWHT")
    if convention not in ("hp", "uhp"):
        raise ValueError(f"unknown convention '{convention}'; have hp, uhp")
    pi = balancing_permutation(eigenvalues, rng)
    inv_pi = np.argsort(pi)
    s1 = rng.choice([-1.0, 1.0], size=d)

    def _h(x: np.ndarray) -> np.ndarray:
        return rot.fwht(x) / np.sqrt(d)

    if convention == "hp":
        def forward(x):
            z = U.T @ x          # U^T: decorrelate into eigen-coordinates
            z = z[pi]            # P:   balancing permutation (eigen-coords)
            z = s1 * z           # D1:  frozen sign flips
            return _h(z)         # H:   normalized Hadamard mix

        def inverse(x):
            h = _h(x)            # H^T = H
            h = s1 * h           # D1
            h = h[inv_pi]        # P^T
            return U @ h         # U:   back to data coordinates

        name = "covHP"
    else:  # "uhp"
        def forward(x):
            z = U.T @ x          # U^T: decorrelate into eigen-coordinates
            z = z[pi]            # P:   balancing permutation (eigen-coords)
            z = s1 * z           # D1:  frozen sign flips
            h = _h(z)            # H:   normalized Hadamard mix
            return U @ h         # U:   map back to the data basis

        def inverse(x):
            y = U.T @ x          # U^T
            z = _h(y)            # H^T = H
            z = s1 * z           # D1
            z = z[inv_pi]        # P^T
            return U @ z         # U:   back to data coordinates

        name = "covUHP"

    return rot.Rotation(forward, inverse, name)


def top_k_variance_fraction(eigenvalues: np.ndarray, k: int) -> float:
    """Fraction of total variance carried by the top-k eigen-directions."""
    return float(np.sum(eigenvalues[:k]) / np.sum(eigenvalues))
