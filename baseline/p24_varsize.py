"""P2.4 — Variable-size residual QJL.

Residual QJL = scalar MSE quantizer (b_mse bits, random-rotation protocol) +
an independent Gaussian sketch of the residual. The sketch dimension m
controls the variance of the inner-product estimator; the scalar bits control
how much of the vector is explained before the sketch must represent it.

Total bit budget: B = b_mse * d + m. Under a fixed B the split is searched by
spending as many bits as possible on the scalar stage (a smaller residual
shrinks the sketch variance) while keeping a sketch of size m:

    b_mse = floor((B - m) / d)

The estimator stays unbiased for every split: E[<y, xhat>] + E[QJL(residual)]
= <y, xhat> + <y, x - xhat> = <y, x> (the QJL term is unbiased for the
residual given the rotation, Lemma 4). b_mse = 0 skips the scalar stage
(pure QJL); m = 0 skips the sketch (pure scalar quantizer).

Pure NumPy. No model, no GPU.
"""
from __future__ import annotations

import numpy as np

from . import codebooks as cb
from . import protocol as pr

__all__ = [
    "split_budget",
    "residual_qjl",
    "residual_qjl_stats",
    "B_MAX_BITS",
]

# Cap on the scalar quantizer bit-width (avoids expensive large codebooks in
# the exact-budget search; never reached at the default budget B = 2d).
B_MAX_BITS = 6


def split_budget(B: int, m: int, d: int) -> int:
    """Best integer scalar bit-width for sketch size m under budget B.

    Spends the remaining budget floor((B - m) / d) on the scalar stage,
    clamped to [0, B_MAX_BITS]; the budget actually used is b_mse*d + m <= B.
    """
    return int(max(0, min(B_MAX_BITS, (B - m) // d)))


def residual_qjl(
    x: np.ndarray,
    y: np.ndarray,
    b_mse: int,
    m: int,
    codebooks: dict,
    rng: np.random.Generator,
) -> float:
    """One-trial residual QJL estimate of <y, x> with sketch size m.

    Protocol: fixed x (adversarial e1); per trial an independent sign-corrected
    Haar rotation for the scalar stage and an independent Gaussian sketch
    S (m x d) of the residual. ``codebooks`` maps b_mse -> codebook (built
    once by the caller); b_mse = 0 skips the scalar stage, m = 0 the sketch.
    """
    d = x.shape[0]
    P = pr.random_rotation(d, rng)
    if b_mse == 0:
        xhat = np.zeros(d)
    else:
        cbk = codebooks[b_mse]
        yx = P @ x
        xhat = P.T @ cb.dequantize(cb.quantize(yx, cbk), cbk)
    if m == 0:
        return float(np.dot(y, xhat))
    r = x - xhat
    S = rng.standard_normal((m, d))
    sk = np.sign(S @ r)
    sq = S @ y
    qjl = np.sqrt(np.pi / 2) / m * np.sum(sq * sk)
    return float(np.dot(y, xhat) + qjl)


def residual_qjl_stats(
    x: np.ndarray,
    y: np.ndarray,
    b_mse: int,
    m: int,
    codebooks: dict,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple:
    """(bias, variance) of the estimator over independent trials."""
    ests = np.empty(n_trials)
    for i in range(n_trials):
        ests[i] = residual_qjl(x, y, b_mse, m, codebooks, rng)
    true = float(np.dot(y, x))
    return float(np.mean(ests) - true), float(np.var(ests))
