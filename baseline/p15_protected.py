"""P1.5 — Protected-token pools (attention sinks) for the KV cache.

Model: a pool of N synthetic unit vectors (tokens). A tiny protected subset
(the first k tokens, standing in for attention sinks / outlier positions) is
quantized at a higher precision b_hi while the remaining N-k tokens are
quantized at a low rate b_lo. The comparison is against uniform quantization
of the whole pool at the SAME total payload budget.

Budget bookkeeping (quantized payload bits only, no metadata):

    B_protected = d * (k * b_hi + (N - k) * b_lo)     bits
    B_uniform   = d * N * b                            bits

The protected pool's average rate is b_eq = (k*b_hi + (N-k)*b_lo)/N, which is
generally fractional, so no integer-bit uniform codebook sits at exactly the
same budget. The matched-budget uniform comparator is therefore reported by
measuring uniform MSE at the bracketing integer rates floor(b_eq) and
ceil(b_eq) and interpolating log(MSE) linearly in rate (scalar-quantizer
distortion is exponential in rate, Zador). All integer-rate values are raw
measurements; only the exact-budget point is interpolated.

Protocol (per the shared rules): a fixed pool (deterministic from the seed),
and per trial an independent sign-corrected Haar rotation per class (protected
/ regular, exactly like the key/value split in P1.2). Each class is quantized
in the rotated basis with its Lloyd-Max codebook (codebooks.py), dequantized
and rotated back; the reported numbers are means of the per-vector squared
error over vectors and trials. Every unit vector has the same
rotation-invariant coordinate law, so the measured per-class MSE depends only
on the bit rate — which token positions are protected does not change the
measured values (the first k are used as the sink set).

Pure NumPy. No model, no GPU.
"""
from __future__ import annotations

import numpy as np

from . import codebooks as cb
from . import protocol as pr

__all__ = [
    "make_pool",
    "class_mse",
    "pool_mse",
    "protected_mse",
    "matched_uniform_mse",
    "benchmark",
]


def make_pool(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Fixed pool of n unit vectors as columns (d, n): iid Gaussian directions."""
    g = rng.standard_normal((d, n))
    return g / np.linalg.norm(g, axis=0, keepdims=True)


def class_mse(
    x_class: np.ndarray,
    codebook: np.ndarray,
    n_rot: int,
    rng: np.random.Generator,
) -> float:
    """Mean per-vector reconstruction MSE of a class under the rotation protocol.

    x_class: (d, n) fixed unit vectors of one class. Per trial one independent
    sign-corrected Haar rotation is applied to the whole class (rotated
    coordinates of every unit vector follow the same Beta law), quantized with
    the class codebook, dequantized and rotated back. Returns the mean
    per-vector squared error over vectors and trials.
    """
    d, n = x_class.shape
    err = 0.0
    for _ in range(n_rot):
        P = pr.random_rotation(d, rng)
        y = P @ x_class
        yhat = cb.dequantize(cb.quantize(y, codebook), codebook)
        xhat = P.T @ yhat
        err += float(np.sum((x_class - xhat) ** 2))
    return err / (n_rot * n)


def pool_mse(
    x_pool: np.ndarray,
    codebook: np.ndarray,
    n_rot: int,
    rng: np.random.Generator,
) -> float:
    """Uniform-rate MSE for the whole pool (same protocol as class_mse)."""
    return class_mse(x_pool, codebook, n_rot, rng)


def protected_mse(
    x_pool: np.ndarray,
    k: int,
    b_lo: int,
    b_hi: int,
    codebooks: dict,
    n_rot: int,
    rng: np.random.Generator,
) -> dict:
    """One (fraction, b_lo, b_hi) protected-pool configuration.

    First k columns are the protected (sink) tokens at b_hi bits; the
    remaining N-k tokens are regular at b_lo bits. Per class, independent
    rotations per trial. Returns measured per-class and aggregate MSE plus the
    exact payload budget (index bits only, no metadata).
    """
    if b_hi <= b_lo:
        raise ValueError(f"need b_hi > b_lo, got ({b_lo}, {b_hi})")
    d, n = x_pool.shape
    x_hi = x_pool[:, :k]
    x_lo = x_pool[:, k:]
    mse_hi = class_mse(x_hi, codebooks[b_hi], n_rot, rng)
    mse_lo = class_mse(x_lo, codebooks[b_lo], n_rot, rng)
    agg = (k * mse_hi + (n - k) * mse_lo) / n
    payload = d * (k * b_hi + (n - k) * b_lo)
    return {
        "k": k,
        "frac": k / n,
        "b_lo": b_lo,
        "b_hi": b_hi,
        "mse_protected": mse_hi,
        "mse_regular": mse_lo,
        "mse_agg": agg,
        "payload_bits": int(payload),
        "avg_bits": payload / (d * n),
    }


def matched_uniform_mse(b_eq: float, uniform: dict) -> dict:
    """Interpolated uniform MSE at fractional rate b_eq (exact budget match).

    ``uniform`` maps integer rate -> measured whole-pool MSE. log(MSE) is
    linearly interpolated between floor(b_eq) and ceil(b_eq) (distortion is
    exponential in rate); the anchors are returned so the reader sees the raw
    measurements. Degenerate (zero) MSE falls back to linear interpolation.
    """
    b0 = int(np.floor(b_eq))
    b1 = int(np.ceil(b_eq))
    m0 = float(uniform[b0])
    m1 = float(uniform[b1])
    if b0 == b1:
        return {"mse_eq": m0, "anchor0": (b0, m0), "anchor1": (b1, m1)}
    w = b_eq - b0
    if m0 <= 0.0 or m1 <= 0.0:
        mse_eq = m0 + (m1 - m0) * w
    else:
        mse_eq = float(np.exp(np.log(m0) * (1.0 - w) + np.log(m1) * w))
    return {"mse_eq": mse_eq, "anchor0": (b0, m0), "anchor1": (b1, m1)}


def benchmark(
    d: int,
    n: int,
    n_rot: int,
    seed: int = 0,
    fracs: tuple = (0.01, 0.05),
    pairs: tuple = ((1, 3), (2, 4)),
) -> dict:
    """Full P1.5 benchmark.

    Builds the fixed pool, measures whole-pool uniform MSE at every integer
    rate up to the largest b_hi used, then every (frac, b_lo, b_hi) protected
    configuration with its matched-budget uniform comparator. Returns all
    measured values (no pass/fail thresholds).
    """
    rng = np.random.default_rng(seed)
    x_pool = make_pool(n, d, rng)

    max_b = max(b_hi for _, b_hi in pairs)
    rates = range(1, max_b + 1)
    codebooks = {b: cb.beta_lloyd_max(d, b) for b in rates}
    uniform = {b: pool_mse(x_pool, codebooks[b], n_rot, rng) for b in rates}

    results = {
        "d": d,
        "n": n,
        "n_rot": n_rot,
        "seed": seed,
        "uniform": uniform,
        "configs": [],
    }
    for frac in fracs:
        for b_lo, b_hi in pairs:
            k = max(1, int(round(frac * n)))
            res = protected_mse(x_pool, k, b_lo, b_hi, codebooks, n_rot, rng)
            m = matched_uniform_mse(res["avg_bits"], uniform)
            res["uniform_mse_eq"] = m["mse_eq"]
            res["anchor0"] = m["anchor0"]
            res["anchor1"] = m["anchor1"]
            res["ratio"] = res["mse_agg"] / m["mse_eq"]
            results["configs"].append(res)
    return results
