"""P2.3 — Orthogonalized residual sketches.

TurboQuant's residual QJL stage (see p24_varsize) estimates <y, x> as a
scalar-stage contribution plus a QJL term on the residual,

    est = <y, xhat> + qjl(r, y),
    qjl(r, y) = scale / m * sum_i sign(<S_i, r>) <S_i, y>,

where xhat is the scalar quantizer's reconstruction of x (b_mse bits under
the random-rotation protocol) and S is an m x d random sketch. This proposal
isolates the sketch-row construction: classic iid Gaussian rows versus rows
made exactly orthonormal by QR-orthogonalizing a Gaussian matrix (an
orthonormal basis of the Gaussian row space). Everything else — the scalar
stage, the rotation protocol, the estimator form — is unchanged.

Unbiasedness. Each row contributes E[sign(<S_i, r>) <S_i, y>] =
<r, y> C / ||r|| with a construction-specific constant C:

- iid rows, S_i ~ N(0, I): C = sqrt(2/pi) (exact joint-Gaussian computation);
- orthonormal rows, S_i uniform on the unit sphere: C = E|u_1| with
  u ~ Unif(S^{d-1}), i.e. C = 2 Gamma(d/2) / ((d-1) sqrt(pi) Gamma((d-1)/2))
  (~ sqrt(2/(pi d)) for large d).

Normalizing the per-trial sketch sum by scale = ||r||/C therefore yields an
unbiased estimator for every residual r in both constructions (the ||r||
factor matters once the scalar stage leaves ||r|| != 1). Orthogonalizing the
rows decorrelates the per-row contributions: E[||Q r||^2] = (m/d) ||r||^2
versus m ||r||^2 for iid rows, which concentrates the sign pattern and is
expected to reduce the estimator variance — measured here.

Protocol: fixed x,y with nonzero dot (x = adversarial e1, y = all_equal),
independent rotation + sketch per trial. Pure NumPy. No SciPy, no model,
no GPU.
"""
from __future__ import annotations

import math

import numpy as np

from . import codebooks as cb
from . import protocol as pr

__all__ = [
    "orthogonal_rows",
    "row_constant",
    "sketch_rows",
    "qjl_term",
    "residual_qjl",
    "residual_qjl_stats",
]


def orthogonal_rows(m: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """m x d sketch matrix with exactly orthonormal rows (QR of a Gaussian).

    Takes a Gaussian m x d matrix G, factors G^T = Q R (reduced QR), and
    returns Q^T: the m columns of Q form an orthonormal basis of the row
    space of G, so the returned rows are orthonormal (each row uniform on the
    sphere). Requires m <= d (at most d orthonormal rows fit in R^d).
    """
    if m > d:
        raise ValueError(f"orthogonalized sketch needs m <= d, got m={m}, d={d}")
    g = rng.standard_normal((m, d))
    q, _ = np.linalg.qr(g.T)  # (d, m): orthonormal columns spanning row(g)
    return q.T                # (m, d): orthonormal rows


def _sphere_abs_cos_expectation(d: int) -> float:
    """E|u_1| for u uniform on the unit sphere in R^d.

    The marginal density of u_1 on [0,1] is 2 c (1 - t^2)^((d-3)/2) with
    c = Gamma(d/2)/(sqrt(pi) Gamma((d-1)/2)); the first moment is
    2 Gamma(d/2) / ((d-1) sqrt(pi) Gamma((d-1)/2)) = Gamma(d/2) /
    (sqrt(pi) Gamma((d+1)/2)).
    """
    return 2.0 * math.gamma(d / 2) / (
        (d - 1) * np.sqrt(np.pi) * math.gamma((d - 1) / 2)
    )


def row_constant(mode: str, d: int) -> float:
    """Per-row constant C with E[sign(<S_i, r>) <S_i, y>] = <r, y> C / ||r||.

    mode "iid": C = sqrt(2/pi) (joint-Gaussian per-row law).
    mode "orth": C = E|u_1|, sphere-marginal per-row law.
    """
    if mode == "iid":
        return np.sqrt(2.0 / np.pi)
    if mode == "orth":
        return _sphere_abs_cos_expectation(d)
    raise ValueError(f"unknown sketch mode '{mode}' (have 'iid', 'orth')")


def sketch_rows(m: int, d: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    """m x d sketch matrix for the residual stage.

    mode "iid": rows iid standard normal.
    mode "orth": QR-orthogonalized rows (requires m <= d).
    """
    if mode == "iid":
        return rng.standard_normal((m, d))
    if mode == "orth":
        return orthogonal_rows(m, d, rng)
    raise ValueError(f"unknown sketch mode '{mode}' (have 'iid', 'orth')")


def qjl_term(
    r: np.ndarray,
    y: np.ndarray,
    sketch: np.ndarray,
    mode: str,
    d: int,
) -> float:
    """Unbiased QJL estimate of <r, y> from an m x d sketch.

    est = ||r|| / (m C) * sum_i sign(<S_i, r>) <S_i, y>, with C the
    per-row constant of the sketch construction (row_constant). The ||r||
    normalization keeps the term unbiased for any residual norm.
    """
    m = sketch.shape[0]
    z = sketch @ r
    w = sketch @ y
    scale = np.linalg.norm(r) / row_constant(mode, d)
    return float(scale / m * np.sum(np.sign(z) * w))


def residual_qjl(
    x: np.ndarray,
    y: np.ndarray,
    b_mse: int,
    m: int,
    mode: str,
    codebooks: dict,
    rng: np.random.Generator,
) -> float:
    """One-trial residual-QJL estimate of <y, x> with sketch rows ``mode``.

    Protocol: fixed x; per trial an independent sign-corrected Haar rotation
    for the scalar stage (b_mse bits, b_mse = 0 skips it) and an independent
    m x d sketch of the residual r = x - xhat with rows per ``mode``.
    ``codebooks`` maps b_mse -> codebook (built once by the caller).
    """
    d = x.shape[0]
    P = pr.random_rotation(d, rng)
    if b_mse == 0:
        xhat = np.zeros(d)
    else:
        cbk = codebooks[b_mse]
        yx = P @ x
        xhat = P.T @ cb.dequantize(cb.quantize(yx, cbk), cbk)
    r = x - xhat
    S = sketch_rows(m, d, mode, rng)
    return float(np.dot(y, xhat) + qjl_term(r, y, S, mode, d))


def residual_qjl_stats(
    x: np.ndarray,
    y: np.ndarray,
    b_mse: int,
    m: int,
    mode: str,
    codebooks: dict,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple:
    """(bias, variance) of the residual-QJL estimator over independent trials."""
    ests = np.empty(n_trials)
    for i in range(n_trials):
        ests[i] = residual_qjl(x, y, b_mse, m, mode, codebooks, rng)
    true = float(np.dot(y, x))
    return float(np.mean(ests) - true), float(np.var(ests))
