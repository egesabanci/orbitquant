"""P2.10 — Predictive residual coding for correlated layer keys.

In an LLM the key vectors of successive layers are correlated, so a direct
b-bit quantizer re-encodes structure the previous layer already revealed.
P2.10 replaces direct quantization of the current layer's key with

    1. a compact linear predictor  p_t = A x_{t-1}^rec  that reconstructs the
       current key from the PREVIOUS LAYER'S RECONSTRUCTED key, and
    2. quantization of only the residual  r_t = x_t - p_t.

Decoding is sequential: layer 0 is quantized directly; every later layer is
reconstructed as  x_t^rec = p_t + Q(r_t). The residual direction is quantized
with the protocol (independent Haar rotation, exact finite-d Beta Lloyd-Max
codebook) and rescaled by the exact residual norm (a per-vector scalar header,
cf. P1.3), so the codebook stays matched to the unit-norm protocol vector.

Predictor families (compact linear):
- scalar: A = alpha*I  — one float; alpha is the fitted layer-to-layer
  correlation. For the isotropic Markov chain this is the optimal linear
  predictor, and rank1 serves as a robustness check.
- rank1:  A = alpha*I + s u v^T  — scalar component plus one rank-1
  correction (2d+1 floats). The correction is the rank-1 truncation of the
  residualized cross-covariance, which is an identity-ridge fit (bounded by
  construction, stable with few calibration pairs).

Calibration: the predictor is trained on the first n_cal layers, using
directly-quantized previous-layer reconstructions (the same inputs it sees at
inference) and full-precision current keys; test layers are never seen during
training. The chain is a synthetic correlated sequence of unit vectors
(Markov with successive correlation rho, started from the protocol's fixed
adversarial e1, then a fixed random isometry makes the coordinates generic).

Reported: per-test-layer direct-quantization MSE vs predictive-residual MSE
(averaged over independent rotations per trial), plus the residual energy
fraction the predictor removes before quantization.

Pure NumPy. No model, no GPU.
"""
from __future__ import annotations

import numpy as np

from . import codebooks as cb
from . import protocol as pr

__all__ = [
    "layer_chain",
    "train_predictor",
    "direct_mse",
    "predictive_mse",
    "measure",
]


def layer_chain(
    d: int, n_layers: int, rho: float, rng: np.random.Generator
) -> np.ndarray:
    """Synthetic correlated unit-norm 'layer keys' (n_layers, d).

    x_0 = e1 (the protocol's fixed adversarial vector); then
    x_t = normalize(rho * x_{t-1} + sqrt(1 - rho^2) * e_t), e_t ~ N(0, I/d),
    so successive layers have inner-product correlation ~rho. A fixed random
    isometry is applied to the whole chain (covariance rho*I is preserved, so
    coordinate-wise correlation survives) to make the coordinates generic.
    """
    layers = np.empty((n_layers, d))
    layers[0] = pr.fixed_vector(d)
    for t in range(1, n_layers):
        e = rng.standard_normal(d) / np.sqrt(d)
        v = rho * layers[t - 1] + np.sqrt(1.0 - rho * rho) * e
        layers[t] = v / np.linalg.norm(v)
    P = pr.random_rotation(d, rng)
    return (P @ layers.T).T


def _quantize_scaled(
    x: np.ndarray, codebook: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Protocol quantizer for an arbitrary (possibly non-unit) vector.

    Quantizes the unit direction x/||x|| under an independent Haar rotation
    with the b-bit Beta codebook and rescales the reconstruction by the exact
    norm (a per-vector scalar header). Zero vector quantizes to zero.
    """
    n = float(np.linalg.norm(x))
    if n == 0.0:
        return np.zeros_like(x)
    u = x / n
    P = pr.random_rotation(u.shape[0], rng)
    uhat = P.T @ cb.dequantize(cb.quantize(P @ u, codebook), codebook)
    return n * uhat


def train_predictor(
    prev_rec: np.ndarray, cur: np.ndarray, rank: int
) -> tuple:
    """Fit a compact linear predictor cur ~ A prev_rec, restricted to rank.

    prev_rec: (n_pairs, d) reconstructed (quantized) previous-layer keys.
    cur: (n_pairs, d) full-precision current-layer keys.
    rank=0: A = alpha*I with alpha = sum <prev,cur> / sum ||prev||^2 (1 float).
    rank=1: A = s u v^T, best rank-1 approximation of the ridge least-squares
        solution (2d floats).
    Returns (predict, meta); predict(x_prev) -> A x_prev, and meta carries
    the storage_bits of the predictor coefficients.
    """
    prev_rec = np.asarray(prev_rec, dtype=np.float64)
    cur = np.asarray(cur, dtype=np.float64)
    n_pairs, d = prev_rec.shape
    if n_pairs == 0:
        raise ValueError("need at least one calibration pair (n_cal >= 2)")
    # scalar component, shared by both predictor families
    alpha = float(np.sum(prev_rec * cur) / np.sum(prev_rec * prev_rec))
    if rank == 0:
        return (lambda x_prev: alpha * x_prev), {
            "rank": 0, "alpha": alpha, "storage_bits": 64,
        }
    if rank == 1:
        # rank-1 correction: residualize the scalar fit, then take the rank-1
        # truncation of the cross-covariance prev_rec^T resid / n_pairs. This
        # is an identity-ridge fit (min ||cur - A prev||^2 + ||A||_F^2), so the
        # correction norm is bounded by ||C|| and cannot amplify.
        resid = cur - alpha * prev_rec
        C = prev_rec.T @ resid / n_pairs
        u, s, vt = np.linalg.svd(C, full_matrices=False)
        s1, u1, v1 = s[0], u[:, 0], vt[0, :]

        def predict(x_prev):
            return alpha * x_prev + (s1 * np.dot(v1, x_prev)) * u1

        return predict, {"rank": 1, "s": s1, "storage_bits": (2 * d + 1) * 64}
    raise ValueError(f"rank must be 0 or 1 (got {rank})")


def direct_mse(
    layers: np.ndarray, codebook: np.ndarray, n_rot: int,
    rng: np.random.Generator, idx,
) -> np.ndarray:
    """Per-vector direct-quantization MSE for layers[idx], over n_rot trials."""
    acc = np.zeros(len(idx))
    for _ in range(n_rot):
        for j, t in enumerate(idx):
            xhat = _quantize_scaled(layers[t], codebook, rng)
            acc[j] += np.sum((layers[t] - xhat) ** 2)
    return acc / n_rot


def predictive_mse(
    layers: np.ndarray, codebook: np.ndarray, n_rot: int,
    rng: np.random.Generator, predict, n_cal: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential residual-coding MSE per test layer, averaged over trials.

    Each trial draws fresh independent Haar rotations for every layer. Layer 0
    is quantized directly; layer t >= 1 is reconstructed as
    x_t^rec = predict(x_{t-1}^rec) + Q(x_t - predict(x_{t-1}^rec)).
    Returns (mse, gain) per test layer (test = layers[n_cal:]); gain is the
    mean squared-norm of the residual before quantization (a fraction of the
    unit key energy, i.e. how much the predictor removes).
    """
    n_layers = layers.shape[0]
    n_test = n_layers - n_cal
    mse = np.zeros(n_test)
    gain = np.zeros(n_test)
    for _ in range(n_rot):
        rec_prev = _quantize_scaled(layers[0], codebook, rng)
        for t in range(1, n_layers):
            pred = predict(rec_prev)
            res = layers[t] - pred
            rec = pred + _quantize_scaled(res, codebook, rng)
            if t >= n_cal:
                j = t - n_cal
                mse[j] += np.sum((layers[t] - rec) ** 2)
                gain[j] += np.sum(res * res)
            rec_prev = rec
    return mse / n_rot, gain / n_rot


def measure(
    d: int,
    n_rot: int,
    n_layers: int,
    n_cal: int,
    rho: float,
    ranks=("scalar", "rank1"),
    b_values=(1, 2),
    seed: int = 0,
) -> dict:
    """Full P2.10 benchmark; see module docstring. Returns a nested dict."""
    if not (1 <= n_cal < n_layers):
        raise ValueError(
            f"need 1 <= n_cal < n_layers (got ncal={n_cal}, nlayers={n_layers})"
        )
    rng = np.random.default_rng(seed)
    layers = layer_chain(d, n_layers, rho, rng)
    out = {"b_values": tuple(int(b) for b in b_values)}
    for b in b_values:
        cbk = cb.beta_lloyd_max(d, b)
        # calibration pairs: (directly-quantized prev, full-precision current)
        prev_rec = np.array(
            [_quantize_scaled(layers[t - 1], cbk, rng) for t in range(1, n_cal)]
        )
        cur = layers[1:n_cal]
        test = list(range(n_cal, n_layers))
        direct = direct_mse(layers, cbk, n_rot, rng, test)
        row = {"direct_mse": float(np.mean(direct)), "direct_per_layer": direct}
        for name in ranks:
            rank = 0 if name == "scalar" else 1
            predict, meta = train_predictor(prev_rec, cur, rank)
            mse, gain = predictive_mse(layers, cbk, n_rot, rng, predict, n_cal)
            row[name] = {
                "mse": float(np.mean(mse)),
                "per_layer": mse,
                "gain": float(np.mean(gain)),
                "storage_bits": meta["storage_bits"],
            }
        out[b] = row
    return out
