"""P1.3 — Quantized norm/radius storage (log-space scalar format).

When a vector is quantized in a rotated basis and reconstructed, the scalar
norm is carried alongside the bitstream as a per-vector header (the "radius"
attention scores are scaled by). P1.3 asks how much it costs to store that
scalar in an 8-bit log-space format instead of full precision.

The format is a uniform quantization of ``log2(norm)`` on ``[lo, hi]`` with
``2**bits`` cells and midpoint reconstruction (pure NumPy, no scipy). For the
protocol's unit vector the stored norm is exactly 1.0: full precision stores it
with zero error, while the 8-bit log grid represents it up to a half-cell
error (about 1% for 8 bits over a 256:1 dynamic range).

We measure:

- relative error of the stored norm: full precision vs 8-bit log, both for the
  nominal unit norm and for the actual per-rotation reconstruction norms;
- propagation to the inner-product estimate: the stored norm is used for
  norm-preserving reconstruction (scale the dequantized vector to the stored
  norm), and we compare the resulting bias ratio against the full-precision
  reference.

Pure NumPy. No model, no GPU.
"""
from __future__ import annotations

import numpy as np

from . import codebooks as cb
from . import protocol as pr

__all__ = [
    "log2_quantize",
    "format_error",
    "measure",
]


def log2_quantize(
    value: float | np.ndarray,
    bits: int = 8,
    lo: float = -4.0,
    hi: float = 4.0,
) -> np.ndarray:
    """Uniform log-space quantization with midpoint reconstruction.

    ``log2(value)`` is uniformly quantized into ``2**bits`` cells spanning
    ``[lo, hi]``; the reconstruction is the cell midpoint, so the maximum
    absolute log2 error is ``(hi - lo) / 2**(bits + 1)`` and out-of-range
    values are clamped to the edge cells. Returns the reconstructed values.
    """
    v = np.asarray(value, dtype=np.float64)
    if np.any(v <= 0):
        raise ValueError("norms must be positive")
    l = np.log2(v)
    step = (hi - lo) / 2.0**bits
    idx = np.clip(np.floor((l - lo) / step), 0.0, 2.0**bits - 1.0)
    return 2.0 ** (lo + (idx + 0.5) * step)


def format_error(bits: int = 8, lo: float = -4.0, hi: float = 4.0) -> tuple[float, float]:
    """Intrinsic resolution of the log format.

    Returns ``(step_log2, worst_relative)``: the log2 width of one cell and
    the worst-case relative error of any representable value (half a cell in
    log space, translated to linear scale).
    """
    step = (hi - lo) / 2.0**bits
    return float(step), float(2.0 ** (step / 2.0) - 1.0)


def measure(
    d: int,
    n_rot: int,
    b: int,
    seed: int = 0,
    bits: int = 8,
    lo: float = -4.0,
    hi: float = 4.0,
) -> dict[str, float]:
    """Benchmark P1.3 with the standard protocol (fixed x, independent rotations).

    - x: fixed adversarial unit vector (e1); y: fixed vector with nonzero dot.
    - Each rotation: quantize ``P @ x`` with the b-bit Beta Lloyd-Max codebook,
      dequantize and rotate back -> ``xraw`` with norm ``s``.
    - Norm-preserving reconstruction scales ``xraw`` by ``stored / s``:
        * full precision: stored = 1.0 (the exact unit norm) -> ``beta_fp``;
        * 8-bit log: stored = ``Q8log(1.0)`` -> ``beta_q`` (constant scale);
        * 8-bit log per-vector: stored = ``Q8log(s)`` -> ``beta_qdist``.
    Returns relative stored-norm errors and the resulting bias ratios.
    """
    rng = np.random.default_rng(seed)
    x = pr.fixed_vector(d)
    y = pr.fixed_vector(d, kind="all_equal")
    cbk = cb.beta_lloyd_max(d, b)
    true = float(np.dot(y, x))

    r_q_unit = float(log2_quantize(np.array([1.0]), bits, lo, hi)[0])

    est_fp = np.empty(n_rot)   # exact stored norm (unit value 1.0)
    est_q = np.empty(n_rot)    # Q8log(1.0) stored, constant across rotations
    est_qd = np.empty(n_rot)   # Q8log(s) stored per rotation
    rel_rec = np.empty(n_rot)  # |Q8log(s) - s| / s for the per-vector header
    for i in range(n_rot):
        P = pr.random_rotation(d, rng)
        xraw = P.T @ cb.dequantize(cb.quantize(P @ x, cbk), cbk)
        s = float(np.linalg.norm(xraw))
        r_q = float(log2_quantize(np.array([s]), bits, lo, hi)[0])
        rel_rec[i] = abs(r_q - s) / s
        inner = np.dot(y, xraw / s)  # full-precision norm-preserved estimate
        est_fp[i] = inner
        est_q[i] = inner * r_q_unit
        est_qd[i] = inner * r_q

    return {
        "r_q_unit": r_q_unit,
        "rel_err_unit": abs(r_q_unit - 1.0),          # format error at norm 1
        "rel_err_rec": float(np.mean(rel_rec)),        # format error on real norms
        "beta_fp": float(np.mean(est_fp) / true),
        "beta_q": float(np.mean(est_q) / true),
        "beta_qdist": float(np.mean(est_qd) / true),
        "check": r_q_unit * float(np.mean(est_fp) / true),
    }
