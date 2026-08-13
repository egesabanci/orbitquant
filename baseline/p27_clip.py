"""P2.7 — Clipped and companded scalar codebooks for heavy tails.

The plain exact-Beta Lloyd-Max codebook (``codebooks.beta_lloyd_max``) is
optimal for the light-tailed rotated-coordinate law (Beta((d-1)/2,(d-1)/2) on
[-1, 1]) that Haar rotations produce. When the coordinate source has heavier
tails than that law assumes, two robustified designs are benchmarked:

1. **Calibrated clipping** — Lloyd-Max on the source density *truncated* to
   [-clip, clip]: ``clipped_lloyd_max``. The clip is calibrated by scanning
   quantiles of |samples| and keeping the clip with the smallest empirical
   MSE. Outliers beyond the clip collapse onto the outermost centroid.

2. **Companding (mu-law / log-law)** — the source is passed through a
   monotone nonlinearity g (mu-law or log-law), quantized uniformly in the
   companded domain, and expanded back: the encoder bins g(x) uniformly and
   the decoder emits the inverse-companded bin midpoints
   (``companded_mse``, the true companding quantizer). The scale s and mu
   (log-law alpha = s/mu) are calibrated the same way. The same levels used
   with a nearest-centroid encoder are also reported (transformed-level
   variant).

Both are compared against the plain exact-Beta codebook at b = 1, 2 and
against the matched Lloyd-Max codebook for the true heavy-tailed law (the
unclipped ideal). Heavy-tailed sources are pure NumPy: an outlier-channel
mixture (98% N(0,1) + 2% N(0,100)) and unit-variance Student-t with df = 3.

Pure NumPy. No model, no GPU.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "P_OUT",
    "SIG_OUT",
    "heavy_tail_pdf",
    "heavy_tail_samples",
    "student_t_pdf",
    "student_t_samples",
    "arcsine_pdf",
    "arcsine_samples",
    "arcsine_lloyd_max",
    "arcsine_clipped_lloyd_max",
    "clipped_lloyd_max",
    "unclipped_lloyd_max",
    "mu_law_codebook",
    "log_law_codebook",
    "companded_mse",
    "mse_samples",
    "calibrate_clip",
    "calibrate_compander",
]

# --------------------------------------------------------------------------- #
# Heavy-tailed coordinate laws
# --------------------------------------------------------------------------- #

# Outlier-channel mixture: bulk N(0, 1) plus P_OUT of N(0, SIG_OUT^2).
P_OUT = 0.02
SIG_OUT = 10.0


def heavy_tail_pdf(x: np.ndarray) -> np.ndarray:
    """Density of the outlier mixture 0.98*N(0,1) + 0.02*N(0,100)."""
    x = np.asarray(x, dtype=np.float64)
    bulk = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    out = np.exp(-0.5 * (x / SIG_OUT) ** 2) / (SIG_OUT * np.sqrt(2.0 * np.pi))
    return (1.0 - P_OUT) * bulk + P_OUT * out


def heavy_tail_samples(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n iid coordinates from the outlier mixture."""
    z = rng.standard_normal(n)
    z = np.where(rng.random(n) < P_OUT, SIG_OUT * z, z)
    return z


def student_t_pdf(x: np.ndarray, df: float = 3.0) -> np.ndarray:
    """Unit-variance Student-t density X = T_df / sqrt(df/(df-2))."""
    x = np.asarray(x, dtype=np.float64)
    scale = np.sqrt(df / (df - 2.0))  # standardize to unit variance
    t = x * scale
    c = math.gamma((df + 1) / 2) / (np.sqrt(df * np.pi) * math.gamma(df / 2))
    return c * (1.0 + t * t / df) ** (-(df + 1) / 2) * scale


def student_t_samples(
    n: int, rng: np.random.Generator, df: float = 3.0
) -> np.ndarray:
    """Draw n iid unit-variance Student-t(df) coordinates."""
    return rng.standard_t(df, size=n) / np.sqrt(df / (df - 2.0))


def arcsine_pdf(x: np.ndarray) -> np.ndarray:
    """Arcsine density 1/(pi*sqrt(1-x^2)) on (-1, 1) (endpoint-safe).

    This is the exact protocol coordinate law at d=2 (Beta(1/2,1/2) scaled to
    [-1,1]) and is genuinely heavy-tailed: U-shaped, infinite density at
    +/-1. Endpoint-safe (zero outside (-1,1)) so grid integration is exact,
    unlike the clipped 1e-16 clamping in ``codebooks.beta_pdf``.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    m = np.abs(x) < 1.0
    out[m] = 1.0 / (np.pi * np.sqrt(1.0 - x[m] * x[m]))
    return out


def arcsine_samples(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n iid arcsine coordinates on [-1, 1] via the inverse CDF.

    F(x) = 1/2 + arcsin(x)/pi  =>  x = -cos(pi*U), U ~ Uniform(0,1).
    """
    u = rng.random(n)
    return -np.cos(np.pi * u)


def _arcsine_lloyd_trunc(
    n_pos: int, tau: float, tol: float = 1e-13, max_iter: int = 500
) -> np.ndarray:
    """Lloyd-Max on the arcsine |X| density truncated to [0, tau], exact.

    Uses the closed-form arcsine integrals (CDF (2/pi)arcsin(x), first moment
    (2/pi)(1-sqrt(1-x^2)), second moment (1/pi)(arcsin(x)-x sqrt(1-x^2))) so
    there is no grid and no endpoint-singularity issue, for any tau <= 1.
    Equal-probability-mass initialization at the quantiles
    sin((j/k) arcsin(tau)); Lloyd fixed-point iteration on exact cells.
    """
    def mass(a: float, b: float) -> float:
        return (2.0 / np.pi) * (np.arcsin(b) - np.arcsin(a))

    def mom1(a: float, b: float) -> float:
        return (2.0 / np.pi) * (np.sqrt(1.0 - a * a) - np.sqrt(1.0 - b * b))

    k = n_pos
    edges = np.empty(k + 1)
    edges[0] = 0.0
    for j in range(1, k):
        edges[j] = np.sin((j / k) * np.arcsin(tau))
    edges[k] = tau
    c = np.empty(k)
    for i in range(k):
        c[i] = mom1(edges[i], edges[i + 1]) / mass(edges[i], edges[i + 1])
    converged = False
    for _ in range(max_iter):
        b = np.empty(k + 1)
        b[0] = 0.0
        b[1:-1] = (c[:-1] + c[1:]) / 2.0
        b[-1] = tau
        c_new = np.empty(k)
        for i in range(k):
            dm = mass(b[i], b[i + 1])
            if dm <= 0:
                sizes = np.array([mass(b[j], b[j + 1]) for j in range(k)])
                jmax = int(np.argmax(sizes))
                c_new[i] = (b[jmax] + b[jmax + 1]) / 2.0
            else:
                c_new[i] = mom1(b[i], b[i + 1]) / dm
        if np.max(np.abs(c_new - c)) < tol:
            c = c_new
            converged = True
            break
        c = c_new
    if not converged:
        raise RuntimeError(
            f"arcsine Lloyd-Max did not converge in {max_iter} iterations "
            f"(k={k}, tau={tau})"
        )
    if not np.all(np.diff(c) > 0):
        raise RuntimeError(f"arcsine Lloyd-Max produced non-sorted codebook: {c}")
    return c


def arcsine_clipped_lloyd_max(b: int, clip: float) -> np.ndarray:
    """b-bit Lloyd-Max codebook on the arcsine law clipped to [-clip, clip].

    Endpoint-safe exact solver (no grid), valid for any clip in (0, 1].
    """
    pos = _arcsine_lloyd_trunc(2 ** (b - 1), clip)
    return np.concatenate([-pos[::-1], pos])


def arcsine_lloyd_max(b: int) -> np.ndarray:
    """Analytic Lloyd-Max codebook for the arcsine law (d=2 protocol).

    Exact truncated solver at clip=1 (the grid Lloyd-Max machinery degenerates
    on the 1/sqrt(1-x^2) spike: ``codebooks.beta_lloyd_max`` does not converge
    at d=2). b=1: +/-2/pi; b=2: +/-0.297, +/-0.854 (agrees with a direct
    boundary scan to 5-6 digits).
    """
    if b not in (1, 2):
        raise ValueError(f"arcsine_lloyd_max defined for b=1,2 (got b={b})")
    return arcsine_clipped_lloyd_max(b, 1.0)


# --------------------------------------------------------------------------- #
# Truncated Lloyd-Max (generalizes codebooks._lloyd_max_symmetric to [0, hi])
# --------------------------------------------------------------------------- #
def _lloyd_max_trunc(
    density: callable,
    n_pos: int,
    hi: float,
    grid: int = 100_001,
    tol: float = 1e-12,
    max_iter: int = 500,
) -> np.ndarray:
    """Symmetric Lloyd-Max on [0, hi]: n_pos positive centroids (ascending).

    Port of ``codebooks._lloyd_max_symmetric`` with the integration domain
    [0, 1] replaced by [0, hi]: equal-probability-mass initialization from the
    truncated density, then Lloyd fixed-point iterations (boundaries at
    midpoints, centroids = conditional means). ``hi`` small gives the clipped
    codebook; ``hi`` large (tails negligible) gives the matched unclipped one.
    """
    t = np.linspace(0.0, hi, grid)
    w = density(t)
    mass = np.cumsum(w) - 0.5 * w - 0.5 * w[0]
    first = np.cumsum(t * w) - 0.5 * (t * w) - 0.5 * (t[0] * w[0])
    M = mass[-1]

    def _idx(v: float) -> int:
        return min(int(np.searchsorted(t, v)), grid - 1)

    k = n_pos
    edges = np.empty(k + 1)
    edges[0] = 0.0
    for j in range(1, k):
        edges[j] = t[np.searchsorted(mass, j / k * M)]
    edges[k] = hi
    c = np.empty(k)
    for i in range(k):
        lo, h = _idx(edges[i]), _idx(edges[i + 1])
        dm = mass[h] - mass[lo]
        c[i] = (first[h] - first[lo]) / dm

    converged = False
    for _ in range(max_iter):
        b = np.empty(k + 1)
        b[0] = 0.0
        b[1:-1] = (c[:-1] + c[1:]) / 2.0
        b[-1] = hi
        c_new = np.empty(k)
        for i in range(k):
            lo, h = _idx(b[i]), _idx(b[i + 1])
            dm = mass[h] - mass[lo]
            if dm <= 0:
                sizes = np.array(
                    [mass[_idx(b[j + 1])] - mass[_idx(b[j])] for j in range(k)]
                )
                jmax = int(np.argmax(sizes))
                c_new[i] = (b[jmax] + b[jmax + 1]) / 2.0
            else:
                c_new[i] = (first[h] - first[lo]) / dm
        if np.max(np.abs(c_new - c)) < tol:
            c = c_new
            converged = True
            break
        c = c_new

    if not converged:
        raise RuntimeError(
            f"Lloyd-Max did not converge in {max_iter} iterations (k={k}, hi={hi})"
        )
    if not np.all(np.diff(c) > 0):
        raise RuntimeError(f"Lloyd-Max produced non-sorted codebook: {c}")
    return c


def clipped_lloyd_max(
    density: callable, b: int, clip: float, **lm_kw
) -> np.ndarray:
    """b-bit Lloyd-Max codebook on the density truncated to [-clip, clip]."""
    n_pos = 2 ** (b - 1)
    pos = _lloyd_max_trunc(density, n_pos, clip, **lm_kw)
    return np.concatenate([-pos[::-1], pos])


def unclipped_lloyd_max(
    density: callable, b: int, hi: float, **lm_kw
) -> np.ndarray:
    """Matched (ideal) Lloyd-Max on the full law; hi covers the tail mass."""
    return clipped_lloyd_max(density, b, hi, **lm_kw)


# --------------------------------------------------------------------------- #
# Companding codebooks (mu-law; log-law is the same family with alpha=s/mu)
# --------------------------------------------------------------------------- #
def mu_law_codebook(b: int, mu: float, scale: float) -> np.ndarray:
    """b-bit mu-law companding codebook on [-scale, scale].

    Cell midpoints m_k in the companded domain [-1, 1] are inverse-companded:
    c_k = sign(m_k) * (scale/mu) * ((1+mu)^|m_k| - 1). mu -> 0 is linear
    (uniform) quantization.
    """
    n = 2**b
    m = -1.0 + (2.0 * np.arange(n) + 1.0) / n
    if mu <= 1e-9:
        return scale * m
    return np.sign(m) * (scale / mu) * (np.power(1.0 + mu, np.abs(m)) - 1.0)


def log_law_codebook(b: int, alpha: float, scale: float) -> np.ndarray:
    """b-bit log-law companding codebook (g(x)=sign(x)ln(1+|x|/alpha)).

    Equivalent to mu-law with mu = scale/alpha; kept for completeness of the
    log/mu-law family.
    """
    n = 2**b
    m = -1.0 + (2.0 * np.arange(n) + 1.0) / n
    return np.sign(m) * alpha * (np.power(1.0 + scale / alpha, np.abs(m)) - 1.0)


def companded_mse(
    samples: np.ndarray, b: int, mu: float, scale: float
) -> float:
    """MSE of the true mu-law companding quantizer on samples.

    Encoder: uniform binning of g(x) in the companded domain [-1, 1]
    (bin edges at -1 + 2k/2^b); decoder: inverse-companded bin midpoints
    c_k = sign(m_k) * (scale/mu) * ((1+mu)^|m_k| - 1). This is the classic
    companding quantizer; note its decision boundaries are NOT the
    nearest-centroid midpoints of the levels (which is the
    transformed-level variant measured via ``mse_samples``).
    """
    s = np.asarray(samples, dtype=np.float64)
    n = 2**b
    if mu <= 1e-9:  # linear (uniform) compander
        g = np.clip(s / scale, -1.0, 1.0)
        c = scale * (-1.0 + (2.0 * np.arange(n) + 1.0) / n)
    else:
        g = np.sign(s) * np.log1p(mu * np.abs(s) / scale) / np.log1p(mu)
        m = -1.0 + (2.0 * np.arange(n) + 1.0) / n
        c = np.sign(m) * (scale / mu) * (np.power(1.0 + mu, np.abs(m)) - 1.0)
    k = np.clip(np.floor((g + 1.0) * n / 2.0).astype(np.int64), 0, n - 1)
    return float(np.mean((s - c[k]) ** 2))


# --------------------------------------------------------------------------- #
# Calibration & evaluation
# --------------------------------------------------------------------------- #
def mse_samples(codebook: np.ndarray, samples: np.ndarray) -> float:
    """Mean squared distance of samples to the nearest codebook centroid."""
    s = np.asarray(samples, dtype=np.float64)
    return float(np.mean(np.min((s[..., np.newaxis] - codebook) ** 2, axis=-1)))


CLIP_QUANTILES = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999]
MU_GRID = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
           10000, 20000]
SCALE_QUANTILES = [0.99, 0.999, 0.9999]


def calibrate_clip(
    density: callable,
    b: int,
    samples: np.ndarray,
    builder: callable | None = None,
    **lm_kw,
) -> tuple:
    """Best (clip, clipped codebook, MSE) over |samples| quantile clips.

    ``builder`` builds the clipped codebook (default ``clipped_lloyd_max``);
    pass an endpoint-safe builder for singular densities, e.g.
    ``arcsine_clipped_lloyd_max`` for the d=2 arcsine law.
    """
    if builder is None:
        builder = clipped_lloyd_max
    absx = np.abs(np.asarray(samples, dtype=np.float64))
    cands = sorted(set(list(np.quantile(absx, CLIP_QUANTILES))
                        + [float(np.max(absx))]))
    best = (None, None, np.inf)
    for tau in cands:
        if tau <= 0.0:
            continue
        cbk = builder(density, b, float(tau), **lm_kw)
        m = mse_samples(cbk, samples)
        if m < best[2]:
            best = (float(tau), cbk, m)
    return best


def calibrate_compander(b: int, samples: np.ndarray) -> tuple:
    """Best (mu, scale, mu-law codebook, MSE) over the (mu, scale) grid.

    The MSE uses the true companding quantizer (uniform bins in the companded
    domain, ``companded_mse``); the returned codebook is the set of
    inverse-companded levels (for the nearest-centroid variant).
    """
    absx = np.abs(np.asarray(samples, dtype=np.float64))
    scale_cands = sorted(set(list(np.quantile(absx, SCALE_QUANTILES))
                              + [float(np.max(absx))]))
    best = (None, None, None, np.inf)
    for mu in MU_GRID:
        for s in scale_cands:
            if s <= 0.0:
                continue
            m = companded_mse(samples, b, mu, float(s))
            if m < best[3]:
                best = (mu, float(s), mu_law_codebook(b, mu, float(s)), m)
    return best
