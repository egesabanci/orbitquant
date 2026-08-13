"""P2.3 — Orthogonalized residual sketches.

The residual-QJL estimator (see p24_varsize) computes <y, x> as the scalar
quantizer contribution plus a QJL term on the residual,

    est = <y, xhat> + sqrt(pi/2) / m * sum_i sign(<S_i, x - xhat>) <S_i, y>,

where S is an m x d random sketch. This proposal isolates the sketch-row
construction: classic iid Gaussian rows versus rows made exactly orthonormal
by QR-orthogonalizing a Gaussian matrix (an orthonormal basis of the Gaussian
row space).

Unbiasedness. Each row contributes E[sign(<S_i, x>) <S_i, y>] =
<x, y> C / ||x|| with a construction-specific constant C:

- iid rows, S_i ~ N(0, I): C = sqrt(2/pi) (exact joint-Gaussian computation);
- orthonormal rows, S_i uniform on the unit sphere: C = E|u_1| with
  u ~ Unif(S^{d-1}), i.e. C = 2 Gamma(d/2) / ((d-1) sqrt(pi) Gamma((d-1)/2))
  (~ sqrt(2/(pi d)) for large d).

Normalizing the per-trial sum by ||x||/C therefore yields an unbiased
estimator in both constructions. Orthogonalizing the rows decorrelates the
per-row contributions: E[||Q x||^2] = (m/d) ||x||^2 versus m ||x||^2 for iid
rows, which concentrates the sign pattern and is expected to reduce the
estimator variance — measured here.

Protocol: fixed x,y with nonzero dot (x = adversarial e1, y = all_equal),
independent sketches per trial. Pure NumPy. No SciPy, no model, no GPU.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "orthogonal_rows",
    "sketch_scale",
    "qjl_estimate",
    "qjl_stats",
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
    2 Gamma(d/2) / ((d-1) sqrt(pi) Gamma((d-1)/2)).
    """
    return 2.0 * math.gamma(d / 2) / (
        (d - 1) * np.sqrt(np.pi) * math.gamma((d - 1) / 2)
    )


def sketch_scale(mode: str, d: int, norm_x: float) -> float:
    """Unbiasedness constant ||x||/C for a sketch-row construction.

    mode "iid": C = sqrt(2/pi) (joint-Gaussian per-row law).
    mode "orth": C = E|u_1|, sphere-marginal per-row law.
    """
    if mode == "iid":
        return norm_x * np.sqrt(np.pi / 2)
    if mode == "orth":
        return norm_x / _sphere_abs_cos_expectation(d)
    raise ValueError(f"unknown sketch mode '{mode}' (have 'iid', 'orth')")


def qjl_estimate(
    x: np.ndarray, y: np.ndarray, sketch: np.ndarray, scale: float
) -> float:
    """One-trial QJL estimate of <y, x> from an m x d sketch.

    est = scale / m * sum_i sign(<S_i, x>) <S_i, y>, with ``scale`` the
    per-mode unbiasedness constant ||x||/C (see sketch_scale).
    """
    m = sketch.shape[0]
    z = sketch @ x
    w = sketch @ y
    return float(scale / m * np.sum(np.sign(z) * w))


def qjl_stats(
    x: np.ndarray,
    y: np.ndarray,
    m: int,
    mode: str,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple:
    """(bias, variance) of the QJL estimator over independent sketches.

    mode "iid": m x d rows iid standard normal.
    mode "orth": QR-orthogonalized rows (orthonormal basis of the Gaussian
    row space). Requires m <= d.
    """
    d = x.shape[0]
    scale = sketch_scale(mode, d, float(np.linalg.norm(x)))
    if mode == "iid":
        def sketch():
            return rng.standard_normal((m, d))
    elif mode == "orth":
        def sketch():
            return orthogonal_rows(m, d, rng)
    else:
        raise ValueError(f"unknown sketch mode '{mode}' (have 'iid', 'orth')")

    ests = np.empty(n_trials)
    for i in range(n_trials):
        ests[i] = qjl_estimate(x, y, sketch(), scale)
    true = float(np.dot(y, x))
    return float(np.mean(ests) - true), float(np.var(ests))
