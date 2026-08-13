"""P2.8 -- OCTOPUS-style triplet codec.

Rotated coordinates are split into contiguous triplets. Each triplet is
stored as its scalar norm plus the direction of the triplet on S^2, mapped to
two octahedral coordinates; the norm and the two direction coordinates are
quantized separately with (possibly different) Lloyd-Max codebooks. This is
the structure of the OCTOPUS k-bit codec (Dettmers et al., ICML 2023) adapted
to the OrbitQuant statistical protocol.

Octahedral mapping (standard octahedral normal encoding, Cigolle et al. 2014):
for a unit direction u = (a, b, c),

    p = u / (|a| + |b| + |c|)            (L1 projection onto the octahedron)
    if p.c < 0: (p.a, p.b) reflected across the diamond edge |a|+|b| = 1
    output (p.a * 0.5 + 0.5, p.b * 0.5 + 0.5) in [0, 1]^2

The fold is an involution, so decoding reconstructs the direction exactly
(verified to machine precision) and every quantized pair in [0, 1]^2 decodes
to a valid unit vector after renormalization.

Bit allocation: with per-coordinate budget b, each triplet gets 3b bits split
as b_norm + 2*b_dir (norm codebook of 2**b_norm centroids, shared direction
codebook of 2**b_dir centroids for both octahedral coordinates). Any split
with b_norm + 2*b_dir == 3b uses exactly b*d total bits, matched to the
b-bit scalar reference; leftover coordinates (d mod 3) are scalar-quantized
with the exact-Beta Lloyd-Max codebook at b bits.

Codebooks: the triplet norm and the octahedral-coordinate marginals have no
closed-form density, so both Lloyd-Max codebooks are solved by empirical
k-means (Lloyd iterations) over calibration samples drawn from the exact
protocol distribution (independent Haar rotations of the fixed vector x).
The leftover coordinates follow the exact Beta coordinate law, so their
codebook is the exact ``beta_lloyd_max`` codebook. Pure NumPy, no scipy.

Protocol: fixed adversarial x = e1, averaged over independent sign-corrected
Haar-QR rotations per trial. MSE is the per-vector squared reconstruction
error (rotation-invariant: ||x - P.T Q(P x)||^2 == ||P x - Q(P x)||^2).
"""
from __future__ import annotations

import numpy as np

from . import codebooks as cb
from . import protocol as pr

__all__ = [
    "octahedral_encode",
    "octahedral_decode",
    "lloyd_max_samples",
    "calibration_samples",
    "codebooks_from_samples",
    "quantize_triplets",
    "triplet_codec_mse",
    "scalar_mse",
    "matched_splits",
]


# --------------------------------------------------------------------------- #
# Octahedral normal encoding (batch, (..., 3) <-> (..., 2))
# --------------------------------------------------------------------------- #
def octahedral_encode(v: np.ndarray) -> np.ndarray:
    """Map unit vectors (..., 3) to octahedral coordinates in [0, 1]^2."""
    v = np.asarray(v, dtype=np.float64)
    l1 = np.abs(v).sum(-1, keepdims=True)
    u = v / l1
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    south = z < 0
    # fold the south hemisphere across the diamond edge |x|+|y| = 1 (both
    # reflected coordinates use the pre-fold values; the map is an involution)
    xf = (1.0 - np.abs(y)) * np.sign(x)
    yf = (1.0 - np.abs(x)) * np.sign(y)
    x = np.where(south, xf, x)
    y = np.where(south, yf, y)
    return np.stack([x * 0.5 + 0.5, y * 0.5 + 0.5], axis=-1)


def octahedral_decode(uv: np.ndarray) -> np.ndarray:
    """Invert :func:`octahedral_encode`; returns unit vectors (..., 3).

    Works for any pair in [0, 1]^2 (centroids may leave the true support):
    z = 1 - |x| - |y| is kept as-is (negative marks the folded hemisphere),
    so out-of-diamond points fold back and the result is renormalized.
    """
    uv = np.asarray(uv, dtype=np.float64)
    x = uv[..., 0] * 2.0 - 1.0
    y = uv[..., 1] * 2.0 - 1.0
    z = 1.0 - np.abs(x) - np.abs(y)
    south = z < 0
    xf = (1.0 - np.abs(y)) * np.sign(x)
    yf = (1.0 - np.abs(x)) * np.sign(y)
    x = np.where(south, xf, x)
    y = np.where(south, yf, y)
    n = np.sqrt(x * x + y * y + z * z)
    return np.stack([x / n, y / n, z / n], axis=-1)


# --------------------------------------------------------------------------- #
# Empirical Lloyd-Max (1-D k-means on protocol-calibrated samples)
# --------------------------------------------------------------------------- #
def lloyd_max_samples(
    samples: np.ndarray,
    k: int,
    max_iter: int = 300,
    tol: float = 1e-12,
) -> np.ndarray:
    """Lloyd-Max codebook of k centroids for a 1-D empirical distribution.

    Initialized from equal-probability-mass quantile cells, then Lloyd
    iterations (assignment by cell midpoint, centroid = cell mean) until the
    centroids stop moving. 1-D k-means always converges finitely; empty cells
    keep their previous centroid (does not happen for quantile inits on
    continuous data). Returns the k centroids ascending.
    """
    s = np.sort(np.asarray(samples, dtype=np.float64).ravel())
    n = s.size
    if k <= 0 or k >= n:
        raise ValueError(f"k={k} out of range for {n} samples")
    # init: k equal-mass cells (quantiles), centroid = cell mean
    edges = np.linspace(0.0, float(n), k + 1)
    lo = np.clip(np.floor(edges[:-1]).astype(int), 0, n - 1)
    hi = np.clip(np.ceil(edges[1:]).astype(int), lo + 1, n)
    c = np.array([s[lo[i]:hi[i]].mean() for i in range(k)])
    for _ in range(max_iter):
        b = np.empty(k + 1)
        b[0], b[-1] = -np.inf, np.inf
        b[1:-1] = 0.5 * (c[:-1] + c[1:])
        idx = np.searchsorted(b[1:-1], s, side="right")
        c_new = np.empty(k)
        for i in range(k):
            sel = s[idx == i]
            c_new[i] = sel.mean() if sel.size else c[i]
        if np.max(np.abs(c_new - c)) < tol:
            c = c_new
            break
        c = c_new
    return c


# --------------------------------------------------------------------------- #
# Codebook construction from the protocol distribution
# --------------------------------------------------------------------------- #
def calibration_samples(
    d: int,
    n_cal: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample triplet norms and octahedral coordinates from the protocol.

    Rotates the fixed vector x = e1 by ``n_cal`` independent Haar rotations;
    for every contiguous triplet collects its norm and both octahedral
    coordinates (pooled -- the two marginals are identical by symmetry).
    Distributions depend only on d, so the samples are shared across all
    bit-allocation splits of the same dimension.
    """
    rng = np.random.default_rng(seed)
    x = pr.fixed_vector(d)
    n_t = d // 3
    norms = np.empty(n_cal * n_t)
    coords = np.empty(2 * n_cal * n_t)
    for i in range(n_cal):
        y = pr.random_rotation(d, rng) @ x
        t = y[: 3 * n_t].reshape(n_t, 3)
        r = np.linalg.norm(t, axis=1)
        norms[i * n_t : (i + 1) * n_t] = r
        uv = octahedral_encode(t / np.maximum(r, 1e-300)[:, None])
        coords[2 * i * n_t : 2 * (i + 1) * n_t] = uv.ravel()
    return norms, coords


def codebooks_from_samples(
    d: int,
    b: int,
    b_norm: int,
    b_dir: int,
    norms: np.ndarray,
    coords: np.ndarray,
) -> dict:
    """Build the three codebooks for a (b_norm, b_dir) allocation.

    Returns {'norm': 2**b_norm, 'coord': 2**b_dir, 'scalar': 2**b} where
    'scalar' is the exact-Beta Lloyd-Max codebook used for the leftover
    coordinates (d mod 3) at the per-coordinate budget b.
    """
    return {
        "norm": lloyd_max_samples(norms, 2**b_norm),
        "coord": lloyd_max_samples(coords, 2**b_dir),
        "scalar": cb.beta_lloyd_max(d, b),
    }


# --------------------------------------------------------------------------- #
# Codec
# --------------------------------------------------------------------------- #
def quantize_triplets(y: np.ndarray, cbs: dict) -> np.ndarray:
    """Reconstruct one rotated vector y from its triplet code (d,).

    Triplets: norm quantized to ``cbs['norm']``, both octahedral coordinates
    to the shared ``cbs['coord']`` codebook; the direction is decoded from the
    quantized coordinates and scaled by the quantized norm. Leftover
    coordinates are scalar-quantized with ``cbs['scalar']``.
    """
    d = y.shape[0]
    n_t = d // 3
    rem = d - 3 * n_t
    t = y[: 3 * n_t].reshape(n_t, 3)
    r = np.linalg.norm(t, axis=1)
    uv = octahedral_encode(t / np.maximum(r, 1e-300)[:, None])
    ri = cb.quantize(r, cbs["norm"])
    ui = cb.quantize(uv, cbs["coord"])
    that = cbs["norm"][ri][:, None] * octahedral_decode(cbs["coord"][ui])
    yhat = that.reshape(-1)
    if rem:
        tail = y[3 * n_t :]
        yhat = np.concatenate([yhat, cbs["scalar"][cb.quantize(tail, cbs["scalar"])]])
    return yhat


def matched_splits(b: int) -> list[tuple[int, int]]:
    """All (b_norm, b_dir) with b_norm + 2*b_dir == 3*b, both >= 1.

    Each split spends exactly 3b bits per triplet, so with b bits on each
    leftover coordinate the total is exactly b*d bits -- matched to the b-bit
    scalar Lloyd-Max reference.
    """
    return [
        (bn, (3 * b - bn) // 2)
        for bn in range(1, 3 * b)
        if (3 * b - bn) % 2 == 0 and (3 * b - bn) // 2 >= 1
    ]


# --------------------------------------------------------------------------- #
# Protocol benchmarks
# --------------------------------------------------------------------------- #
def triplet_codec_mse(
    d: int,
    n_rot: int,
    cbs: dict,
    rng: np.random.Generator,
) -> float:
    """Mean per-vector MSE of the triplet codec over independent rotations."""
    x = pr.fixed_vector(d)
    mse = 0.0
    for _ in range(n_rot):
        y = pr.random_rotation(d, rng) @ x
        mse += float(np.sum((y - quantize_triplets(y, cbs)) ** 2))
    return mse / n_rot


def scalar_mse(d: int, n_rot: int, b: int, rng: np.random.Generator) -> float:
    """Mean per-vector MSE of the b-bit scalar Beta Lloyd-Max reference."""
    x = pr.fixed_vector(d)
    cbk = cb.beta_lloyd_max(d, b)
    mse = 0.0
    for _ in range(n_rot):
        y = pr.random_rotation(d, rng) @ x
        mse += float(np.sum((y - cb.dequantize(cb.quantize(y, cbk), cbk)) ** 2))
    return mse / n_rot
