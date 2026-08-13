"""P1.6 — Adaptive bit allocation by measured per-coordinate variance.

Rate-distortion scheduler over the scalar Lloyd-Max quantizers: per-coordinate
variance is measured over a calibration pass of independent rotations of the
fixed adversarial vector, and a fixed total bit budget is distributed across
coordinates so that every bit goes to the coordinate with the largest marginal
distortion gain (greedy water-filling on the measured rate-distortion curve).

Protocol (shared): fixed x = e1, independent sign-corrected Haar rotations per
trial. For Haar rotations every rotated coordinate has the same marginal law
(the exact Beta(d) codebook law, variance 1/d), so the measured variances are
statistically equal and the variance-driven allocation is expected to
degenerate to uniform allocation -- the benchmark measures the actual gap at a
matched total budget. A synthetic outlier-variance profile is included to
demonstrate the allocator mechanism when coordinate variances genuinely differ
(the P1.6 story: "allocate bits where they buy the most quality").

Pure NumPy. No model, no GPU.
"""
from __future__ import annotations

import heapq

import numpy as np

from . import codebooks as cb
from . import protocol as pr

__all__ = [
    "unit_distortion",
    "distortion_curve",
    "measured_coordinate_variance",
    "allocate_bits",
    "uniform_mse",
    "adaptive_mse",
    "synthetic_outlier_eval",
]

B_MIN_BITS = 1
# Cap on per-coordinate bit-width: the shared exact-Beta Lloyd-Max solver
# (codebooks._lloyd_max_symmetric) does not converge beyond b=4 at d=64.
B_MAX_BITS = 4


# --------------------------------------------------------------------------- #
# Rate-distortion curve of the scalar codebooks
# --------------------------------------------------------------------------- #
def unit_distortion(d: int, b: int, grid: int = 200_001) -> float:
    """MSE of the exact Beta(d) Lloyd-Max codebook, per unit variance.

    For Z with the rotated-coordinate law f_X (codebook law, variance 1/d),
    returns E[(Z - Q_b(Z))^2] / E[Z^2] -- the distortion of b-bit
    quantization normalized so that a coordinate with measured variance
    sigma^2 and b_j bits has expected per-coordinate MSE sigma^2 * D_rel(b_j).
    Computed by trapezoid integration over the exact density (pure NumPy).
    """
    t = np.linspace(-1.0, 1.0, grid)
    f = cb.beta_pdf(t, d)
    cbk = cb.beta_lloyd_max(d, b)
    q = cbk[np.argmin(np.abs(t[:, None] - cbk[None, :]), axis=1)]
    h = 2.0 / (grid - 1)
    w = np.ones(grid)
    w[0] = w[-1] = 0.5
    mass = float(np.sum(w * f) * h)
    m2 = float(np.sum(w * f * t * t) * h)
    dist = float(np.sum(w * f * (t - q) ** 2) * h)
    return dist / m2


def distortion_curve(d: int, b_min: int = B_MIN_BITS, b_max: int = B_MAX_BITS) -> np.ndarray:
    """Per-unit-variance distortion D_rel(b) for b in [b_min, b_max]."""
    return np.array([unit_distortion(d, b) for b in range(b_min, b_max + 1)])


# --------------------------------------------------------------------------- #
# Calibration: per-coordinate variance over independent rotations
# --------------------------------------------------------------------------- #
def measured_coordinate_variance(
    d: int, n_calib: int, rng: np.random.Generator
) -> np.ndarray:
    """Per-coordinate sample variance of the rotated fixed vector.

    Rotates x = e1 with n_calib independent Haar rotations and takes the
    sample variance of each coordinate across the calibration pass.
    """
    x = pr.fixed_vector(d)
    cols = np.empty((n_calib, d))
    for i in range(n_calib):
        P = pr.random_rotation(d, rng)
        cols[i] = P @ x
    return cols.var(axis=0, ddof=1)


# --------------------------------------------------------------------------- #
# Bit allocation (greedy marginal gain / water-filling)
# --------------------------------------------------------------------------- #
def allocate_bits(
    variances: np.ndarray,
    drel: np.ndarray,
    budget: int,
    b_min: int = B_MIN_BITS,
    b_max: int = B_MAX_BITS,
) -> np.ndarray:
    """Distribute ``budget`` bits to minimize sum_j var_j * D_rel(b_j).

    Greedy water-filling: start every coordinate at b_min, then repeatedly give
    the next bit to the coordinate with the largest marginal distortion gain
    var_j * (D_rel(b_j) - D_rel(b_j + 1)). This is exact for the separable
    integer problem because D_rel is convex decreasing in b (marginal gains of
    each coordinate are non-increasing). Requires d*b_min <= budget <= d*b_max.
    """
    variances = np.asarray(variances, dtype=float)
    d = variances.shape[0]
    if len(drel) != b_max - b_min + 1:
        raise ValueError("drel must cover b_min..b_max")
    if budget < d * b_min:
        raise ValueError(f"budget {budget} < d*b_min = {d * b_min}")
    if budget > d * b_max:
        raise ValueError(f"budget {budget} > d*b_max = {d * b_max}")

    b = np.full(d, b_min, dtype=int)
    used = d * b_min

    def gain(j: int) -> float:
        return float(variances[j] * (drel[b[j] - b_min] - drel[b[j] - b_min + 1]))

    heap = [(-gain(j), j, b_min) for j in range(d)]
    heapq.heapify(heap)
    while used < budget:
        g, j, bj = heapq.heappop(heap)
        if b[j] != bj:  # stale entry for a coordinate already incremented
            continue
        b[j] += 1
        used += 1
        if b[j] < b_max:
            heapq.heappush(heap, (-gain(j), j, b[j]))
    return b


# --------------------------------------------------------------------------- #
# Evaluation (protocol: fixed e1, independent Haar rotations per trial)
# --------------------------------------------------------------------------- #
def uniform_mse(
    d: int, n_rot: int, b: int, codebook: np.ndarray, rng: np.random.Generator
) -> float:
    """Per-vector MSE of uniform b-bit quantization over n_rot rotations."""
    x = pr.fixed_vector(d)
    total = 0.0
    for _ in range(n_rot):
        P = pr.random_rotation(d, rng)
        y = P @ x
        xhat = P.T @ cb.dequantize(cb.quantize(y, codebook), codebook)
        total += float(np.sum((x - xhat) ** 2))
    return total / n_rot


def adaptive_mse(
    d: int,
    n_rot: int,
    allocation: np.ndarray,
    codebooks: dict,
    rng: np.random.Generator,
) -> float:
    """Per-vector MSE of per-coordinate bit allocation over n_rot rotations.

    Coordinate j of the rotated vector is quantized with codebooks[allocation[j]]
    (codebooks: bit-width -> Lloyd-Max codebook, built once by the caller).
    """
    x = pr.fixed_vector(d)
    total = 0.0
    for _ in range(n_rot):
        P = pr.random_rotation(d, rng)
        y = P @ x
        yhat = np.empty_like(y)
        for b in np.unique(allocation):
            cbk = codebooks[b]
            m = allocation == b
            yhat[m] = cb.dequantize(cb.quantize(y[m], cbk), cbk)
        xhat = P.T @ yhat
        total += float(np.sum((x - xhat) ** 2))
    return total / n_rot


# --------------------------------------------------------------------------- #
# Mechanism check: allocation under a genuine variance structure
# --------------------------------------------------------------------------- #
def synthetic_outlier_eval(
    d: int,
    n_trials: int,
    drel: np.ndarray,
    n_outliers: int = 4,
    var_mult: float = 10.0,
    b_avg: int = 2,
    seed: int = 0,
) -> dict:
    """Uniform vs adaptive MSE for coordinates with outlier variances.

    Synthetic coordinate model (no rotation): normal coordinates have variance
    1/d, ``n_outliers`` have variance var_mult/d. Coordinates are drawn iid
    Gaussian; coordinate j is quantized with the Beta codebook scaled to its
    variance sqrt(d*var_j) * beta_lloyd_max(d, b). Reports measured MSE of the
    uniform (b_avg) and adaptive (water-filled, same total budget) allocations
    plus the adaptive bit distribution. Demonstrates that the allocator
    concentrates bits on high-variance coordinates.
    """
    rng = np.random.default_rng(seed)
    var = np.full(d, 1.0 / d)
    var[:n_outliers] *= var_mult
    budget = b_avg * d
    alloc = allocate_bits(var, drel, budget)
    b_u = budget // d
    codebooks = {b: cb.beta_lloyd_max(d, b) for b in np.unique(np.concatenate([alloc, [b_u]]))}
    scale = np.sqrt(d * var)

    def _quantize_block(z: np.ndarray, bits: np.ndarray) -> np.ndarray:
        zhat = np.empty_like(z)
        for b in np.unique(bits):
            cbk = codebooks[b]
            m = bits == b
            zhat[m] = cb.dequantize(cb.quantize(z[m] / scale[m], cbk), cbk) * scale[m]
        return zhat

    uni, ada = 0.0, 0.0
    for _ in range(n_trials):
        z = rng.standard_normal(d) * scale
        uni += float(np.sum((z - _quantize_block(z, np.full(d, b_u))) ** 2))
        ada += float(np.sum((z - _quantize_block(z, alloc)) ** 2))
    n = float(n_trials)
    return {
        "n_outliers": n_outliers,
        "var_mult": var_mult,
        "budget": budget,
        "b_uniform": b_u,
        "allocation": alloc,
        "uniform_mse": uni / n,
        "adaptive_mse": ada / n,
    }
