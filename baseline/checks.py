"""Statistical checks against TurboQuant's theoretical claims.

Each check verifies one claim from the paper. All are deterministic given a
fixed seed.

Protocol (matches the paper's claim, lines 507-514): the claim is that for a
FIXED (even worst-case) vector x, averaging over independent random rotations
Pi makes Pi*x's coordinates follow the exact Beta law. So every check fixes
x (e.g. the adversarial basis vector e_1) and averages over many independent,
sign-corrected Haar-QR rotations. This is what genuinely discriminates Haar
from a bad structured transform: for fixed x = e_1, a single Hadamard gives
all coordinates +/-1/sqrt(d) (not Beta), while Haar gives Beta.

Checks:
  1. coordinate_distribution -- Lemma 1: fixed x, random Pi -> coordinates Beta
  2. near_independence       -- the crux: distinct rotated coords near-independent
  3. mse_distortion          -- Eq.4/Thm 1: measured MSE matches theory
  4. inner_product_bias      -- Sec 3.2: b=1 E[<y,Q(x)>] = (2/pi)<y,x>
  5. qjl                     -- Lemma 4: unbiased; var <= pi/(2d)||y||^2
  6. near_optimality         -- Thm 3: within ~2.7x of Shannon floor 1/4^b

Every check returns (name, measured, expected, tolerance, passed).
"""
from __future__ import annotations

import numpy as np

from . import codebooks as cb

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """Independent Haar random rotation (sign-corrected QR of a Gaussian).

    QR of a Gaussian matrix is Haar up to column signs; fixing the sign of each
    column by the sign of the corresponding R diagonal yields the unique QR and
    a Haar-distributed orthogonal matrix (diagonal treated as +1 when zero).
    """
    a = rng.standard_normal((d, d))
    q, r = np.linalg.qr(a)
    q *= np.where(np.diag(r) >= 0, 1.0, -1.0)
    return q


def fixed_vector(d: int, kind: str = "e1") -> np.ndarray:
    """A fixed, adversarial unit vector (worst-case input)."""
    x = np.zeros(d)
    if kind == "e1":
        x[0] = 1.0
    elif kind == "all_equal":
        x[:] = 1.0 / np.sqrt(d)
    else:
        raise ValueError(kind)
    return x


def _beta_cdf(d: int, samples: np.ndarray, grid: int = 200_001) -> np.ndarray:
    """Exact CDF of the coordinate law (TurboQuant Lemma 1) by grid integration.

    Pure NumPy: integrates the closed-form beta_pdf on a fine grid and
    interpolates at the sample points (matches the exact Beta CDF to ~6e-10).
    """
    t = np.linspace(-1.0, 1.0, grid)
    w = cb.beta_pdf(t, d)
    cdf = np.cumsum(w) - 0.5 * w - 0.5 * w[0]
    cdf = cdf / cdf[-1]
    return np.interp(samples, t, cdf)


def _beta_ks(d: int, samples: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic vs the exact Beta cdf."""
    x = np.sort(samples)
    cdf = _beta_cdf(d, x)
    n = len(x)
    ecdf = np.arange(1, n + 1) / n
    return float(np.max(np.abs(ecdf - cdf)))


# --------------------------------------------------------------------------- #
# The six checks
# --------------------------------------------------------------------------- #
def check_coordinate_distribution(d: int, n_rot: int, rng: np.random.Generator):
    """Lemma 1: fixed x, independent random Pi -> coordinates follow Beta."""
    x = fixed_vector(d)
    samples = np.empty(n_rot * d)
    for i in range(n_rot):
        P = random_rotation(d, rng)
        samples[i * d : (i + 1) * d] = P @ x
    ks = _beta_ks(d, samples)
    return ("coordinate_distribution", ks, 0.02, ks < 0.02)


def check_near_independence(d: int, n_rot: int, rng: np.random.Generator):
    """The crux: distinct rotated coordinates are near-independent.

    For fixed x, average over rotations; probe with the correlation of
    coordinates and a higher-order moment-factorization check.
    """
    x = fixed_vector(d)
    n = n_rot * d
    y = np.empty((n_rot, d))
    for i in range(n_rot):
        y[i] = random_rotation(d, rng) @ x
    corr = np.corrcoef(y, rowvar=False)
    off = corr - np.eye(d)
    # RMS off-diagonal correlation (stable metric; max is inflated by sampling
    # noise ~ 2-3/sqrt(nrot) for many coordinates)
    rms_off = float(np.sqrt(np.mean(off**2)))
    # higher-order: E[c1^2 c2^2] ~ E[c1^2] E[c2^2]
    c1, c2 = y[:, 0], y[:, 1]
    lhs = np.mean(c1**2 * c2**2)
    rhs = np.mean(c1**2) * np.mean(c2**2)
    factor_gap = float(abs(lhs - rhs) / (rhs + 1e-12))
    return ("near_independence", rms_off, 0.03, rms_off < 0.03 and factor_gap < 0.1)


def check_mse_distortion(d: int, b: int, n_rot: int, rng: np.random.Generator):
    """Eq. 4 / Thm 1: measured MSE matches the codebook's theoretical value."""
    x = fixed_vector(d)
    cbk = cb.beta_lloyd_max(d, b)
    mse = 0.0
    for _ in range(n_rot):
        P = random_rotation(d, rng)
        y = P @ x
        idx = cb.quantize(y, cbk)
        yhat = cb.dequantize(idx, cbk)
        mse += float(np.sum((y - yhat) ** 2))
    mse /= n_rot
    ref = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}[b]
    # paper values are per-vector (sum over d coordinates); the exact-Beta
    # codebook matches them closely at both d=64 and d=128.
    return ("mse_distortion", mse, ref, abs(mse - ref) < 0.05)


def check_inner_product_bias(d: int, b: int, n_rot: int, rng: np.random.Generator):
    """Sec 3.2: at b=1, E[<y, Q(x)>] = (2/pi) <y,x>.

    Fixed x,y with nonzero dot; average over independent rotations Pi. The
    bias is the regression slope of <y, Q(x)> on <y,x> across trials.
    """
    x = fixed_vector(d)
    y = fixed_vector(d, kind="all_equal")  # <y,x> = 1/sqrt(d) != 0
    true = float(np.dot(y, x))
    cbk = cb.beta_lloyd_max(d, b)
    ests = np.empty(n_rot)
    for i in range(n_rot):
        P = random_rotation(d, rng)
        yx = P @ x
        idx = cb.quantize(yx, cbk)
        xhat = P.T @ cb.dequantize(idx, cbk)
        ests[i] = np.dot(y, xhat)
    # regression slope = E[est]/true (true fixed, nonzero)
    ratio = float(np.mean(ests) / true)
    return ("inner_product_bias", ratio, 2 / np.pi, abs(ratio - 2 / np.pi) < 0.05)


def check_qjl(d: int, m: int, n_trials: int, rng: np.random.Generator):
    """Lemma 4: QJL unbiased; variance <= pi/(2d)||y||^2.

    Fixed x,y with nonzero dot; independent sketches S per trial; variance
    around the fixed true score.
    """
    x = fixed_vector(d)
    y = fixed_vector(d, kind="all_equal")
    true = float(np.dot(y, x))
    ests = np.empty(n_trials)
    for i in range(n_trials):
        S = rng.standard_normal((m, d))
        sk = np.sign(S @ x)
        sq = S @ y
        ests[i] = np.sqrt(np.pi / 2) / m * np.sum(sq * sk)
    bias = float(np.mean(ests) - true)
    var = float(np.var(ests))
    bound = (np.pi / 2) / m * np.dot(y, y)
    return ("qjl", (bias, var), (0.0, bound), abs(bias) < 0.05 and var <= bound * 1.2)


def check_near_optimality(d: int, b: int, n_rot: int, rng: np.random.Generator):
    """Thm 3: measured MSE vs the Shannon floor 1/4^b (within ~2.7x)."""
    x = fixed_vector(d)
    cbk = cb.beta_lloyd_max(d, b)
    mse = 0.0
    for _ in range(n_rot):
        P = random_rotation(d, rng)
        y = P @ x
        idx = cb.quantize(y, cbk)
        yhat = cb.dequantize(idx, cbk)
        mse += float(np.sum((y - yhat) ** 2))
    mse /= n_rot
    floor = 1.0 / (4**b)
    gap = mse / floor
    return ("near_optimality", gap, np.sqrt(3 * np.pi / 2), gap < np.sqrt(3 * np.pi / 2) * 1.1)


ALL_CHECKS = {
    "coordinate_distribution": check_coordinate_distribution,
    "near_independence": check_near_independence,
    "mse_distortion": check_mse_distortion,
    "inner_product_bias": check_inner_product_bias,
    "qjl": check_qjl,
    "near_optimality": check_near_optimality,
}
