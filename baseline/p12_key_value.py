"""P1.2 — Asymmetric key/value bit budgets for the KV cache.

Keys and values play different roles in attention:

- Keys are scored by inner product <y, x>, so they need LOW inner-product
  error. Keys are quantized with the b-bit product estimator (TurboQuant
  Algorithm 2): a (K-1)-bit MSE base codebook plus a 1-bit QJL residual,
  total K bits per coordinate.
- Values are averaged by attention, so they need GOOD RECONSTRUCTION. Values
  are quantized with the V-bit Lloyd-Max MSE codebook, V bits per coordinate.

This module benchmarks asymmetric splits (K, V) at the same average budget
(K+V)/2 -- e.g. (4,2) vs (2,4) vs (3,3) -- using the standard protocol
(fixed x,y with nonzero dot, independent sign-corrected Haar rotations, pure
NumPy). Reported per split: key inner-product error (bias + RMSE), value MSE,
and total bits.

Pure NumPy. No model, no GPU.
"""
from __future__ import annotations

import numpy as np

from . import codebooks as cb
from . import protocol as pr

__all__ = [
    "key_product_estimate",
    "value_reconstruct",
    "benchmark_split",
]


def key_product_estimate(
    x: np.ndarray,
    y: np.ndarray,
    rotation: np.ndarray,
    base_codebook: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """b-bit product estimator of <y, x> for one rotation.

    TurboQuant Algorithm 2: (K-1)-bit MSE base quantization of the rotated
    vector, rotated back, plus the 1-bit QJL residual sketch. Total key budget
    is K bits per coordinate (base (K-1) + residual 1).
    """
    d = x.shape[0]
    yx = rotation @ x
    xhat = rotation.T @ cb.dequantize(cb.quantize(yx, base_codebook), base_codebook)
    r = x - xhat
    gamma = np.linalg.norm(r)
    S = rng.standard_normal((d, d))  # independent QJL sketch per trial
    est = np.dot(y, xhat) + gamma * np.sqrt(np.pi / 2) / d * np.sum(
        (S @ y) * np.sign(S @ r)
    )
    return float(est)


def value_reconstruct(
    x: np.ndarray,
    rotation: np.ndarray,
    codebook: np.ndarray,
) -> np.ndarray:
    """Reconstruct x through the V-bit MSE codebook in the rotated basis."""
    yx = rotation @ x
    idx = cb.quantize(yx, codebook)
    return rotation.T @ cb.dequantize(idx, codebook)


def benchmark_split(
    d: int,
    k_bits: int,
    v_bits: int,
    n_rot: int,
    rng: np.random.Generator,
) -> dict:
    """Run one (K, V) split with the standard protocol.

    Fixed adversarial x (e1) and fixed y (all-equal) with nonzero dot, averaged
    over n_rot independent rotations. Keys get a fresh rotation per trial and
    values get another (independent) one. Returns measured statistics only.
    """
    if k_bits < 2:
        raise ValueError(f"key budget K must be >= 2 (base needs K-1 >= 1 bits), got {k_bits}")
    if v_bits < 1:
        raise ValueError(f"value budget V must be >= 1, got {v_bits}")

    x = pr.fixed_vector(d)
    y = pr.fixed_vector(d, kind="all_equal")
    true = float(np.dot(y, x))

    key_cbk = cb.beta_lloyd_max(d, k_bits - 1)  # MSE base at K-1 bits
    val_cbk = cb.beta_lloyd_max(d, v_bits)  # MSE codebook at V bits

    prod_ests = np.empty(n_rot)
    val_mse = np.empty(n_rot)
    for i in range(n_rot):
        Pk = pr.random_rotation(d, rng)
        prod_ests[i] = key_product_estimate(x, y, Pk, key_cbk, rng)
        Pv = pr.random_rotation(d, rng)
        vhat = value_reconstruct(x, Pv, val_cbk)
        val_mse[i] = float(np.sum((x - vhat) ** 2))

    return {
        "K": k_bits,
        "V": v_bits,
        "total_bits": d * (k_bits + v_bits),
        "key_bias": float(np.mean(prod_ests) - true),
        "key_bias_ratio": float(np.mean(prod_ests) / true),
        "key_rmse": float(np.sqrt(np.mean((prod_ests - true) ** 2))),
        "value_mse": float(np.mean(val_mse)),
        "value_floor": 1.0 / (4**v_bits),
    }
