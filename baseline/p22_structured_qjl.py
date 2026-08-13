"""P2.2 -- Structured residual QJL (SRHT/FWHT residual sketch).

The b-bit product estimator (TurboQuant Algorithm 2, P1.2) is an ``b_mse``-bit
Lloyd-Max MSE quantizer of the Haar-rotated vector, rotated back, plus a 1-bit
QJL residual sketch that recovers the lost inner product <y, r> with
r = x - xhat:

    est = <y, xhat> + ||r|| * C / m * sum_j sign(s_j . r) * (s_j . y)

where s_1..s_m are the sketch rows. This module swaps the dense Gaussian
residual sketch for a structured SRHT/FWHT sketch and measures both.

Sketch variants (both calibrated to be exactly unbiased for the protocol's
fixed adversarial x = e1, unit norm):

- dense:      rows s_j iid N(0, I_d), C = sqrt(pi/2).
              E[s sign(s . r)] = sqrt(2/pi) r/||r||  (Gaussian identity), so
              the ||r||-normalized term is unbiased for *any* r.
- structured: SRHT rows s_j = (1/sqrt(d)) h_j (.) D, where D is a random
              +/-1 sign vector drawn per trial and h_j is a Hadamard row
              (subsampled m of d rows); applied via the FWHT butterfly,
              O(d log d) instead of O(m d). C = sqrt(d).
              For a Rademacher row, E[sign(s . r) (s . y)] = c(r) <r, y> with
              c(r) = E|s . r| / ||r||^2. For r = e1, c(r) = E|s_1| = 1, so the
              structured sketch is *exactly* unbiased on the protocol's
              adversarial x; for general residuals c(r) ~= 1 (about
              sqrt(2/pi) for a spread-out residual) and a small residual bias
              (c(r) - 1)<y, r> is measured. Note: at m = d the full FWHT
              sketch is deterministic for e1 (the sign pattern telescopes),
              giving zero-variance estimation.

Fixed x,y with nonzero dot, independent sketch per trial (protocol). Pure
NumPy; no SciPy, no model, no GPU.
"""
from __future__ import annotations

import numpy as np

from . import codebooks as cb
from . import protocol as pr
from . import rotations as rot

__all__ = [
    "DENSE_CONST",
    "structured_constant",
    "product_estimate",
    "product_stats",
]


DENSE_CONST = np.sqrt(np.pi / 2)


def structured_constant(d: int) -> float:
    """Calibration constant for the SRHT/FWHT sketch at dimension d.

    Rows are (1/sqrt(d)) h (.) D, so E[sign(s . r) (s . y)] = c(r) <r, y>/sqrt(d);
    multiplying the mean of m terms by sqrt(d) gives c(r) <r, y> -- exactly
    <r, y> when the sketched vector is e1 (c = 1, the protocol's x).
    """
    return float(np.sqrt(d))


def _structured_rows(d: int, m: int, rng: np.random.Generator) -> tuple:
    """One SRHT sketch: random sign vector D and m subsampled row indices."""
    signs = rng.choice([-1.0, 1.0], size=d)
    rows = np.sort(rng.choice(d, size=m, replace=False))
    return signs, rows


def _srht_coords(
    x: np.ndarray, signs: np.ndarray, rows: np.ndarray, d: int
) -> np.ndarray:
    """Subsampled coordinates of (1/sqrt(d)) H D x via the FWHT butterfly."""
    z = rot.fwht(signs * x) / np.sqrt(d)
    return z[rows]


def product_estimate(
    x: np.ndarray,
    y: np.ndarray,
    b_mse: int,
    m: int,
    codebooks: dict,
    rng: np.random.Generator,
    sketch: str = "dense",
) -> float:
    """One-trial b-bit product estimate of <y, x> with residual sketch type.

    Protocol: fixed x (adversarial e1); per trial an independent sign-corrected
    Haar rotation for the scalar stage and an independent sketch (dense
    Gaussian rows or SRHT/FWHT rows) of the residual r = x - xhat.
    ``codebooks`` maps b_mse -> codebook (built once by the caller);
    b_mse = 0 skips the scalar stage (pure QJL on x), m = 0 skips the sketch.
    ``sketch`` is "dense" or "structured".
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
    gamma = np.linalg.norm(r)
    if sketch == "dense":
        S = rng.standard_normal((m, d))
        term = np.sum((S @ y) * np.sign(S @ r))
        const = DENSE_CONST
    elif sketch == "structured":
        signs, rows = _structured_rows(d, m, rng)
        z = _srht_coords(r, signs, rows, d)
        w = _srht_coords(y, signs, rows, d)
        term = np.sum(np.sign(z) * w)
        const = structured_constant(d)
    else:
        raise ValueError(f"unknown sketch {sketch!r}; have 'dense', 'structured'")
    return float(np.dot(y, xhat) + gamma * const / m * term)


def product_stats(
    x: np.ndarray,
    y: np.ndarray,
    b_mse: int,
    m: int,
    codebooks: dict,
    n_trials: int,
    rng: np.random.Generator,
    sketch: str = "dense",
) -> tuple:
    """(bias, variance) of the estimator over independent trials."""
    ests = np.empty(n_trials)
    for i in range(n_trials):
        ests[i] = product_estimate(x, y, b_mse, m, codebooks, rng, sketch)
    true = float(np.dot(y, x))
    return float(np.mean(ests) - true), float(np.var(ests))
