"""P2.6 — Importance-weighted bit allocation.

Under a fixed total bit budget, coordinates that carry more energy are
quantized with more bits. Two honest readings are benchmarked:

A. Fixed allocation from a calibration pass: per-coordinate sensitivity is
   estimated by rotating the fixed adversarial vector x and averaging the
   squared rotated magnitudes over the calibration rotations. The allocation
   pattern is computed ONCE from this sensitivity vector and applied to every
   trial; no side information is transmitted. Under independent Haar rotations
   all coordinates are exchangeable, so the calibration cannot single out
   fixed important coordinates and this reduces to (near-)uniform allocation
   — measured, not assumed.

B. Per-vector adaptive allocation with explicit signaling cost: the allocation
   is recomputed per trial from the realized importance w_i = y_i^2 (squared
   rotated magnitudes) using the greedy rule on the distortion model

       D(b) = sum_i w_i * 2^(-2 b_i)

   Each bit goes to the coordinate with the largest current priority
   w_i * 4^(-b_i), i.e. the largest marginal distortion reduction. The
   per-vector pattern must be signaled to the decoder; it is charged against
   the budget at S = d * ceil(log2(bmax+1)) bits (naive per-coordinate
   encoding, worst case). Uniform and adaptive are then compared at matched
   TOTAL bits: S + quantizer bits == uniform bits.

Pure NumPy. No model, no GPU.
"""
from __future__ import annotations

import numpy as np

from . import codebooks as cb
from . import protocol as pr

__all__ = [
    "calibrate_sensitivity",
    "allocate_bits",
    "quantize_allocated",
    "uniform_mse",
    "fixed_allocation_mse",
    "allocation_stats",
]


def calibrate_sensitivity(
    x: np.ndarray,
    n_cal: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Calibration pass: per-coordinate sensitivity over n_cal rotations of x.

    Sensitivity of coordinate i is measured as the mean squared rotated
    magnitude E[(P x)_i^2] over independent sign-corrected Haar rotations.
    Under the protocol all coordinates are exchangeable, so this is ~1/d
    uniformly; the calibration validates the sensitivity model, and the
    allocator applies per-vector weights w_i = y_i^2 at quantization time.
    """
    d = x.shape[0]
    s = np.zeros(d)
    for _ in range(n_cal):
        y = pr.random_rotation(d, rng) @ x
        s += y * y
    return s / n_cal


def allocate_bits(
    importance: np.ndarray,
    budget: int,
    b_max: int,
) -> np.ndarray:
    """Greedy integer bit allocation under a fixed total budget.

    Distortion model D(b) = sum_i w_i * 2^(-2 b_i); the marginal reduction of
    upgrading coordinate i from b_i to b_i + 1 bits is (3/4) * w_i * 2^(-2
    b_i), so each bit is assigned to the argmax of w_i * 4^(-b_i). Returns an
    int array of per-coordinate bit widths (0..b_max) summing to ``budget``.
    """
    d = importance.shape[0]
    if budget < 0:
        raise ValueError(f"budget must be >= 0 (got {budget})")
    if budget > b_max * d:
        raise ValueError(
            f"budget {budget} exceeds capacity b_max*d = {b_max}*{d}"
        )
    bits = np.zeros(d, dtype=np.int64)
    for _ in range(budget):
        prio = importance * np.power(0.25, bits)
        prio[bits >= b_max] = -np.inf
        i = int(np.argmax(prio))
        bits[i] += 1
    return bits


def quantize_allocated(
    y: np.ndarray,
    bits: np.ndarray,
    codebooks: dict,
) -> np.ndarray:
    """Per-coordinate quantization with per-coordinate codebooks.

    Coordinate i is quantized with the beta Lloyd-Max codebook for bits[i]
    bits; 0-bit coordinates are reconstructed as 0.
    """
    yhat = np.zeros_like(y)
    for b, cbk in codebooks.items():
        m = bits == b
        if m.any():
            yhat[m] = cb.dequantize(cb.quantize(y[m], cbk), cbk)
    return yhat


def uniform_mse(
    x: np.ndarray,
    b: int,
    codebook: np.ndarray,
    n_rot: int,
    rng: np.random.Generator,
) -> float:
    """Mean per-vector MSE of uniform b-bit quantization (matched budget b*d)."""
    mses = np.empty(n_rot)
    for i in range(n_rot):
        mses[i] = pr.mse_after_quantize(x, codebook, pr.random_rotation(x.shape[0], rng))
    return float(np.mean(mses))


def fixed_allocation_mse(
    x: np.ndarray,
    bits: np.ndarray,
    codebooks: dict,
    n_rot: int,
    rng: np.random.Generator,
) -> float:
    """Mean per-vector MSE with a FIXED per-coordinate bit pattern.

    The pattern ``bits`` (computed once by the caller from the calibration
    pass) is applied to every trial's rotated vector; nothing is signaled per
    vector, so the comparison with uniform allocation is at exactly the same
    bit cost.
    """
    d = x.shape[0]
    mses = np.empty(n_rot)
    for i in range(n_rot):
        P = pr.random_rotation(d, rng)
        y = P @ x
        xhat = P.T @ quantize_allocated(y, bits, codebooks)
        mses[i] = float(np.sum((x - xhat) ** 2))
    return float(np.mean(mses))


def allocation_stats(
    x: np.ndarray,
    budget: int,
    b_max: int,
    codebooks: dict,
    n_rot: int,
    rng: np.random.Generator,
) -> tuple:
    """(mean MSE, mean bit distribution) with per-vector adaptive allocation.

    Per trial: rotate x, estimate importance w = y^2, allocate ``budget``
    quantizer bits greedily, quantize per-coordinate, rotate back.
    ``codebooks`` maps b -> codebook for b in 1..b_max. The bit distribution
    is the mean number of coordinates receiving each bit width, averaged over
    trials. NOTE: the per-vector allocation pattern costs S =
    d*ceil(log2(b_max+1)) signaling bits; the caller charges S against the
    total budget when comparing with uniform allocation.
    """
    d = x.shape[0]
    mses = np.empty(n_rot)
    counts = np.zeros(b_max + 1, dtype=np.float64)
    for i in range(n_rot):
        P = pr.random_rotation(d, rng)
        y = P @ x
        bits = allocate_bits(y * y, budget, b_max)
        xhat = P.T @ quantize_allocated(y, bits, codebooks)
        mses[i] = float(np.sum((x - xhat) ** 2))
        counts += np.bincount(bits, minlength=b_max + 1)
    return float(np.mean(mses)), counts / n_rot
