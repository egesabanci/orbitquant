"""benchmark_tq.py -- honest rate-distortion and estimator benchmark (autoresearch).

Purpose
-------
Primary research-loop benchmark for OrbitQuant's statistical (pure-NumPy,
no-model, no-GPU) regime. Measures the CURRENT code on:

  * tq_mse      : mean per-vector reconstruction MSE over a fixed grid of
                  honest configs. LOWER is better (the "better" axis --
                  primary metric of the loop).
  * tq_bytes_b2 : honest true bits/token (P0.5 accounting) at b=2/d=64 with an
                  explicit 6-bit log norm header. LOWER is better (the
                  "lightweight" axis).
  * tq_bias_b1  : |mean inner-product bias ratio - 2/pi| at b=1. LOWER is
                  better (estimator / attention-logit fidelity).
  * tq_bias_raw_b1 : |mean inner-product bias ratio - 1.0| at b=1 -- how far
                  the estimator is from UNBIASED (1.0 = faithful logits).
                  LOWER is better. SURROGATE diagnostic only: exp 11 showed
                  multiplicative debias (ratio->1) HURTS softmax attention
                  (noise amplification), so DEFAULT_DEBIAS is False and this
                  metric must NOT be optimized on its own.
  * tq_bias_raw  : mean |inner-product bias ratio - 1.0| over the grid
                  (d,b) in {64,128}x{1,2} -- representative estimator
                  unbiasedness surrogate (same anti-debias caveat; raw grid
                  ~0.21, debiased ~0.05).
  * tq_attn_kl_b2 : softmax-KL over a realistic key+query set under ONE
                  shared serving rotation at b=2/d=64 (the REAL attention
                  objective, lower better). Added exp 11; the metric that
                  refuted the multiplicative debias default. Guardrail metric.
  * tq_var_b2   : product-estimator variance / Theorem-2 bound at total b=2.
                  LOWER is better (estimator fidelity).
  * fwht_us     : best-of-N wall-clock to apply the 3-round FWHT structured
                  rotation at d=128. LOWER is better (compute lightness).

Anti-overfitting / anti-cheating contract
-----------------------------------------
1. Realistic inputs: direction is uniform on the sphere, magnitude is
   log-normal (KV-like norms), so norm side information is REQUIRED. A
   representation that silently assumes unit norms pays norm reconstruction
   error on these inputs and scores worse -- dropping the norm header cannot
   be gamed into a win.
2. Fixed honest rates on the quality grid: payload = b*d plus an explicit
   6-bit log norm header, mirrored by P0.5 accounting (reported bytes).
   Configurations never exceed their stated budget.
3. Outputs average over a grid spanning d in {64, 128} and b in {1, 2}, so no
   single configuration can be overfit.
4. All randomness is seeded; identical inputs every run -- the only thing that
   can change the numbers is the code under test (never the benchmark).

Some quantities (exact-Beta Lloyd-Max scalar quantization on Beta coordinates)
are THEORETICALLY near-optimal (TurboQuant Thm 1), so tq_mse headroom is
expected to be small and honest; the real headroom lives in side-info bytes
(tq_bytes_b2), estimator fidelity (tq_bias_b1, tq_var_b2) and compute
(fwht_us). Manufacturing tq_mse gains by editing this benchmark or the seeds
is a violation.

Usage:
    python -m baseline.benchmark_tq            # print full report + METRIC lines
    python -m baseline.benchmark_tq --selfcheck  # invariants (used by checks.sh)
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import codebooks as cb
from . import p05_accounting as p05
from . import protocol as pr
from . import rotations as rot

__all__ = [
    "realistic_dataset",
    "quantize_log_norm",
    "bnorm",
    "grid_mse",
    "true_bits_b2",
    "bias_err_b1",
    "product_var_ratio_b2",
    "fwht_apply_us",
    "run",
    "selfcheck",
]

SEED = 20260816
N_VEC = 1200          # dataset vectors per config
N_ROT = 4             # independent rotations averaged per config (quality grid)
N_BIAS = 800          # rotations for the b=1 bias estimate
N_VAR = 600           # trials for the product-estimator variance

# Adopted (exp 06/08) then REVERTED (exp 11): the finite-d P1.7 multiplicative
# debias set the b=1/b=2 inner-product ratio -> 1.0 (surrogate tq_bias_raw), but
# a realistic attention-objective probe (softmax-KL over a realistic key set
# under ONE shared serving rotation, exp 11) shows it HURTS attention: a
# multiplicative fix amplifies per-key logit NOISE variance by 1/c^2 (~x2.4 at
# b=1) and softmax attention is noise-limited, not scale-limited (raw c~0.89 at
# b=2 is already near temperature-correct). Reverted to False; tq_bias_raw is
# retained as an honest (surrogate) diagnostic only.
DEFAULT_DEBIAS = False

# Side-info norm header: 6-bit log format (P1.3-style), adopted in exp 03.
# Measured tradeoff on realistic lognormal(0, 0.35) norms: 8-bit -> 0.295923
# MSE @ 136 b/token; 6-bit -> 0.296047 (+0.04%) @ 134 b/token. 4-bit ->
# +0.49% MSE @ 132 b/token. The header is 100% of the side-info overhead at
# b=2/d=64, so 6 bits is the honest sweet spot.
# The fixed log-range is calibrated to [-1.2, 1.2] (exp 09): at sigma=0.35 the
# fixed [-1,1] clipped the ~0.2% ln-norm tails; widening to 1.2 improves
# tq_mse 0.296047 -> 0.295584 (fully recovers the 6-bit adoption cost, lands
# BELOW the original 8-bit baseline) at the same 134 b/token. Wider than 1.2
# coarsens the bulk bins again (1.5 -> 0.295622).
NORM_BITS = 6
LOG_RANGE = 1.2


# --------------------------------------------------------------------------- #
# Dataset (realistic: uniform direction, log-normal magnitude)
# --------------------------------------------------------------------------- #
def realistic_dataset(
    d: int, n: int, seed: int, log_sigma: float = 0.35
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, u, r): x = r*u with u uniform on S^{d-1}, r log-normal.

    Log-norm sigma 0.35 => norms ~exp(N(0,0.35)) (most in [0.7, 1.4]), so the
    norm header is genuinely informative and cannot be dropped for free.
    """
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = np.exp(rng.standard_normal(n) * log_sigma)
    return u * r[:, None], u, r


def quantize_log_norm(r: np.ndarray, bits: int,
                      lo: float = -LOG_RANGE, hi: float = LOG_RANGE) -> np.ndarray:
    """Quantize a norm to a `bits`-bit log format over [lo, hi]; returns r_hat."""
    L = 2 ** bits
    t = (np.log(r) - lo) / (hi - lo)
    idx = np.minimum(((t.clip(0.0, 1.0)) * L).astype(np.int64), L - 1)
    qlog = lo + (idx + 0.5) * (hi - lo) / L
    return np.exp(qlog)


def bnorm(norm_bits: int) -> float:
    """Honest per-token side-info bits for a `norm_bits`-bit log norm header."""
    return float(norm_bits)


# --------------------------------------------------------------------------- #
# Quality grid: per-vector MSE at fixed honest rates
# --------------------------------------------------------------------------- #
def grid_mse(d_list=(64, 128), b_list=(1, 2), n: int = N_VEC,
             nrot: int = N_ROT, norm_bits: int = NORM_BITS,
             seed: int = SEED) -> dict:
    """Per-config and aggregate per-vector MSE at fixed honest rates.

    For each (d, b): encode each vector's direction with the exact-Beta
    b-bit Lloyd-Max codebook under independent Haar rotations, store the norm
    with a `norm_bits`-bit log header, reconstruct each vector as
    xhat = r_hat * u_hat, and measure the true E||x - xhat||^2 (norm error
    included). Returns {'per': {(d,b): mse}, 'mean': aggregate}.
    """
    per = {}
    for d in d_list:
        x, u, r = realistic_dataset(d, n, seed ^ d)
        rhat = quantize_log_norm(r, norm_bits)
        for b in b_list:
            rng = np.random.default_rng(seed ^ d ^ b)
            cbk = cb.beta_lloyd_max(d, b)
            acc = 0.0
            for _ in range(nrot):
                P = pr.random_rotation(d, rng)
                y = u @ P.T
                yhat = cb.dequantize(cb.quantize(y, cbk), cbk)
                uhat = yhat @ P
                xhat = rhat[:, None] * uhat
                acc += float(np.mean(np.sum((x - xhat) ** 2, axis=1)))
            per[(d, b)] = acc / nrot
    mean = float(np.mean(list(per.values())))
    return {"per": per, "mean": mean}


def true_bits_b2(d: int = 64, side: float | None = None,
                 n_tokens: int = 8192) -> float:
    """Honest true bits/token (P0.5 accounting) for b=2/d=64 + side-info.

    ``side`` defaults to the adopted 6-bit log norm header (NORM_BITS).
    """
    if side is None:
        side = bnorm(NORM_BITS)
    rep = p05.Representation(
        name="b2",
        payload_bits_per_coord=2.0,
        per_token_side_info_bits=side,
        block_size=64,
    )
    return float(p05.accounting(rep, d=d, n_tokens=n_tokens)["total_bits_per_token"])


# --------------------------------------------------------------------------- #
# Estimator fidelity
# --------------------------------------------------------------------------- #
def bias_err_b1(d: int = 64, nrot: int = N_BIAS, seed: int = SEED) -> float:
    """|mean inner-product bias ratio - 2/pi| at b=1 (paper Sec 3.2)."""
    cbk = cb.beta_lloyd_max(d, 1)
    ratio = pr.inner_product_bias_ratio(
        pr.fixed_vector(d), pr.fixed_vector(d, kind="all_equal"), cbk,
        nrot, np.random.default_rng(seed))
    return float(abs(ratio - 2.0 / np.pi))


def estimator_ratio(d: int, b: int, nrot: int, seed: int = SEED,
                    cal_rot: int = 500) -> float:
    """Mean inner-product ratio E[<y, xhat>] / <y, x> at bit-width b through
    the code's CURRENT scoring path.

    When DEFAULT_DEBIAS, the reconstruction is scaled by the finite-d P1.7
    correction for (d, b) (reciprocal of the codebook's measured bias ratio),
    making the estimator unbiased (ratio -> 1.0) at zero per-token byte cost.
    The raw ratios are far from 1 even at b=2 (d=64/b=2 raw ~0.86, i.e. ~14%
    logit shrinkage), which is exactly why the debias generalizes beyond b=1.
    """
    x = pr.fixed_vector(d)
    y = pr.fixed_vector(d, kind="all_equal")
    cbk = cb.beta_lloyd_max(d, b)
    true = float(np.dot(y, x))
    if DEFAULT_DEBIAS:
        from . import corrections as corr
        factor = corr.bias_correction_factor(d, b, n_rot=cal_rot, seed=seed)
    else:
        factor = 1.0
    rng = np.random.default_rng(seed ^ 0x5A5A)
    ests = np.empty(nrot)
    for i in range(nrot):
        P = pr.random_rotation(d, rng)
        yx = P @ x
        idx = cb.quantize(yx, cbk)
        xhat = P.T @ cb.dequantize(idx, cbk)
        ests[i] = np.dot(y, xhat) / factor
    return float(np.mean(ests) / true)


def estimator_ratio_b1(d: int = 64, nrot: int = N_BIAS, seed: int = SEED) -> float:
    """b=1 specialization (back-compat wrapper)."""
    return estimator_ratio(d, 1, nrot, seed)


def bias_raw_err_b1(d: int = 64, nrot: int = N_BIAS, seed: int = SEED) -> float:
    """|mean inner-product bias ratio - 1.0| at b=1 (faithful-logits error)."""
    ratio = estimator_ratio_b1(d, nrot, seed)
    return float(abs(ratio - 1.0))


# grid configs over which the representative estimator-fidelity metric is averaged
GRID_BIAS_CFGS = ((64, 1), (64, 2), (128, 1), (128, 2))


def attention_metrics(d: int, b: int, n_db: int = 1500, n_q: int = 30,
                      seed: int = SEED, debias: bool | None = None,
                      rot: str = "haar", rot_seed: int | None = None) -> dict:
    """Attention-objective fidelity of the code's current MSE estimator.

    The REAL serving situation: ONE shared rotation P encodes the whole key
    cache (encoder side), then queries score the stored codes in the rotated
    domain. This is *not* the per-pair Monte Carlo over many P: it is a single
    P with per-key quantization noise, which is what actual attention sees.

    ``rot``: "haar" (ideal, the leaderboard protocol) or "hdhdh" (the fast
    3-round structured rotation, exp 02; its coordinates are only nearly-Beta,
    and production would use ONE FIXED hdhdh sign pattern). ``rot_seed`` sets
    the serving rotation; when None it derives from ``seed``. Returns also the
    per-key score-noise std / mean-|true-score| (the noise-limited regime's
    decisive quantity, exp 11/12).

    Pipeline per key (adopted design): store 6-bit log norm rhat (+1.2
    range), quantize the unit direction u with the exact-Beta b-bit Lloyd-Max
    codebook in the rotated domain; score = rhat * <Py, Q(Pu)> (debias factor
    per (d,b) when debias, matching DEFAULT_DEBIAS).

    Returns: softmax-KL (mean over queries), recall@k (ordering fidelity), and
    the tail metrics (p95 of |logit error| / mean-|true-logit| ratio and the
    worst-decile KL) -- because attention risk is typically tail-limited.
    All deterministic given seed (keys, queries, shared P, RR-free).
    """
    if debias is None:
        debias = DEFAULT_DEBIAS
    rot_seed = rot_seed if rot_seed is not None else seed
    if rot == "haar":
        P = pr.random_rotation(d, np.random.default_rng(rot_seed))
    else:
        from . import rotations as rotmod
        R = rotmod.hadamard_sign_flip(d, np.random.default_rng(rot_seed), rounds=3)
        Id = np.eye(d)
        P = np.column_stack([R.forward(Id[:, k]) for k in range(d)])  # P @ x = R(x)
    X, U, R = realistic_dataset(d, n_db, seed ^ 0xA77)  # realistic keys
    Qy, Uy, Ry = realistic_dataset(d, n_q, seed ^ 0xB77)  # realistic queries
    rhat = quantize_log_norm(R, NORM_BITS)
    cbk = cb.beta_lloyd_max(d, b)
    # encode cache in the rotated domain
    W = U @ P.T                     # rotated unit directions (n_db, d)
    QW = cb.dequantize(cb.quantize(W, cbk), cbk)  # centroid codes in rotated dom
    factor = 1.0
    if debias:
        from . import corrections as corr
        factor = corr.bias_correction_factor(d, b, n_rot=500, seed=seed)
    kl = 0.0
    r1 = r5 = r10 = 0.0
    lr95 = []
    noise = 0.0
    for j in range(n_q):
        Py = Qy[j] @ P.T           # rotated query (vector)
        s_true = X @ Qy[j]         # (n_db,)
        s_est = rhat * (QW @ Py) / factor   # (n_db,) with queried norm in Py
        # per-key score-noise (normalized by mean-|true-score|)
        scale = float(np.mean(np.abs(s_true)))
        noise += float(np.std(s_est - s_true) / max(scale, 1e-9))
        # softmax KL
        p = np.exp(s_true - np.max(s_true)); p /= p.sum()
        q = np.exp(s_est - np.max(s_est)); q /= q.sum()
        kl += float(np.sum(p * np.log(p / np.clip(q, 1e-12, None))))
        # recall@k
        k_top_true = np.argsort(s_true)[-10:]
        k_top_est = np.argsort(s_est)[-10:]
        set_e = set(k_top_est)
        r1 += 1.0 if k_top_true[-1] in set_e else 0.0
        r5 += len(set(k_top_true[-5:]) & set_e) / 5.0
        r10 += len(set(k_top_true) & set_e) / 10.0
        # tail: p95 of |logit error| relative to mean-|true-logit|
        scale = float(np.mean(np.abs(s_true)))
        lr95.append(float(np.percentile(np.abs(s_est - s_true), 95) / max(scale, 1e-9)))
    n = float(n_q)
    return {
        "kl": kl / n,
        "recall1": r1 / n,
        "recall5": r5 / n,
        "recall10": r10 / n,
        "p95_logit_ratio": float(np.mean(lr95)),
        "score_noise": noise / n,
    }


def attn_kl_b2(d: int = 64, debias: bool | None = None) -> float:
    """softmax-KL of the adopted estimator vs full precision at b=2/d=64
    (lower better) -- the representative attention-objective guardrail."""
    return float(attention_metrics(d, 2, debias=debias)["kl"])


def attn_kl_b2_p95(d: int = 64, n_q: int = 120, seed: int = SEED) -> float:
    """p95 of the PER-QUERY softmax-KL at b=2/d=64 (exp 17).

    The mean (tq_attn_kl_b2) is tail-inflated: per-query KL is heavily
    right-skewed (p99 ~ 7.6x the mean) and driven by query norm (corr ~+0.94;
    sharp attention amplifies per-key score noise). This tail metric is the
    risk-relevant guardrail: it must stay small for long-context serving.
    Lower better. Deterministic given seed.
    """
    return float(attention_kl_tail(d, b=2, n_q=n_q, seed=seed)["p95"])


def attention_frontier(d: int = 64, n_db: int = 1500, n_q: int = 40,
                       seed: int = SEED) -> list[dict]:
    """Attention-objective rate-distortion frontier (exp 12).

    For each honest representation computes (true bits/token, softmax-KL,
    recall@5) over a realistic key+query set under ONE shared serving rotation
    P (and one shared QJL sketch S for the prod path) -- the deployment choice
    question: what does each byte budget buy on the REAL objective?

    Configs: scalar b={1,2,3,4} (bits = b*d + 6-bit norm), scalar b=2 with
    8-bit norm (128+8, norm-bits sensitivity at the attention objective), and
    prod b=2 (TurboQuant Algorithm 2: 1-bit MSE base + m=64 QJL sign bits +
    6-bit key norm + 6-bit residual norm = 140 bits; residual norm gamma is
    quantized to the same 6-bit log format over ln-range [-2, 0] for honest
    bytes).

    Returns a list of dicts {"label", "bits", "kl", "recall5"} sorted by bits.
    All deterministic given seed.
    """
    rng = np.random.default_rng(seed)
    P = pr.random_rotation(d, rng)
    X, U, R = realistic_dataset(d, n_db, seed ^ 0xA77)
    Qy, _, _ = realistic_dataset(d, n_q, seed ^ 0xB77)
    S = rng.standard_normal((d, d))     # shared QJL sketch (prod path only)
    Sy = S @ Qy.T                        # (d, n_q): sketch of original-domain queries

    def _ks(s_true, s_est):
        p = np.exp(s_true - s_true.max()); p /= p.sum()
        q = np.exp(s_est - s_est.max()); q /= q.sum()
        return float(np.sum(p * np.log(p / np.clip(q, 1e-12, None))))

    def _rc5(s_true, s_est):
        return len(set(np.argsort(s_true)[-5:]) & set(np.argsort(s_est)[-5:])) / 5.0

    rows = []

    def add(label, bits, scorer):
        kl_acc = r5 = 0.0
        for j in range(n_q):
            y = Qy[j]
            s_true = X @ y
            s_est = scorer(y @ P.T, y, j)
            kl_acc += _ks(s_true, s_est)
            r5 += _rc5(s_true, s_est)
        rows.append({"label": label, "bits": float(bits),
                     "kl": kl_acc / n_q, "recall5": r5 / n_q})

    # scalar MSE estimator across rates (6-bit log norm header, exp-03 design)
    W = U @ P.T
    rhat = quantize_log_norm(R, NORM_BITS)
    for b in (1, 2, 3, 4):
        cbk = cb.beta_lloyd_max(d, b)
        QW = cb.dequantize(cb.quantize(W, cbk), cbk)
        add(f"scalar b={b}", b * d + NORM_BITS,
            lambda Py, y, j, QW=QW, rhat=rhat: rhat * (QW @ Py))

    # norm-bits sensitivity at the attention objective (8-bit header)
    cbk2 = cb.beta_lloyd_max(d, 2)
    rhat8 = quantize_log_norm(R, 8)
    QW2 = cb.dequantize(cb.quantize(W, cbk2), cbk2)
    add("scalar b=2, 8-bit norm", 2 * d + 8,
        lambda Py, y, j, QW=QW2, rhat=rhat8: rhat * (QW @ Py))

    # prod b=2 (TurboQuant Algorithm 2): 1-bit MSE base + m=64 QJL residual
    cbk1 = cb.beta_lloyd_max(d, 1)
    QW1 = cb.dequantize(cb.quantize(W, cbk1), cbk1)      # rotated-domain 1-bit base
    xhat_dir = QW1 @ P                                   # original-domain direction
    resid = (R[:, None] * U) - rhat[:, None] * xhat_dir  # original-domain residual
    gamma = np.linalg.norm(resid, axis=1)
    ghat = quantize_log_norm(gamma, NORM_BITS, lo=-2.0, hi=0.0)  # honest 6-bit gamma
    sk = np.sign(resid @ S.T)                            # (n_db, m) sign bits

    def prod_scorer(Py, y, j):
        base_s = rhat * (QW1 @ Py)
        qjl = ghat * np.sqrt(np.pi / 2) / d * (sk @ Sy[:, j])
        return base_s + qjl

    add("prod b=2 (1bit+QJL m=64)", 2 * d + 2 * NORM_BITS, prod_scorer)

    rows.sort(key=lambda r: r["bits"])
    return rows


def kv_attention_error(d: int = 64, b_k: int = 2, b_v: int = 2,
                       n_db: int = 1500, n_q: int = 40,
                       seed: int = SEED, est_p: bool = True) -> dict:
    """Attention-output (value-side) error of a K/V representation pair (exp 15).

    Values are consumed in attention as the weighted average out = sum_i p_i v_i,
    so the value objective is the error of that average under the quantized
    pipeline -- NOT per-vector MSE. With realistic keys and values (independent
    draws of the same law), separate shared serving rotations P_k / P_v, the
    adopted key pipeline (6-bit norm + b_k-bit direction codes) and value
    pipeline (6-bit norm + b_v-bit codes), computes:

      rel_out_err = E_q || sum_i p_i v_i - sum_i p_hat_i vhat_i ||^2
                    / E_q || sum_i p_i v_i ||^2

    where p are the TRUE attention weights and p_hat the estimated ones
    (when est_p=True the p_hat path is end-to-end -- key quantization also
    perturbs the weights; when est_p=False only the values are quantized,
    isolating the value-side contribution). Also returns the softmax-KL of
    p_hat vs p (key-side quality) and the true bits/token
    = (b_k + b_v)*d + 2*NORM_BITS. Deterministic given seed.
    """
    rng = np.random.default_rng(seed)
    Pk = pr.random_rotation(d, rng)
    Pv = pr.random_rotation(d, rng)
    Xk, Uk, Rk = realistic_dataset(d, n_db, seed ^ 0xA77)
    Xv, Uv, Rv = realistic_dataset(d, n_db, seed ^ 0xC77)
    Qy, _, _ = realistic_dataset(d, n_q, seed ^ 0xB77)
    rhat_k = quantize_log_norm(Rk, NORM_BITS)
    rhat_v = quantize_log_norm(Rv, NORM_BITS)
    cbk_k = cb.beta_lloyd_max(d, b_k)
    cbk_v = cb.beta_lloyd_max(d, b_v)
    Wk = Uk @ Pk.T
    QWk = cb.dequantize(cb.quantize(Wk, cbk_k), cbk_k)
    Wv = Uv @ Pv.T
    QWv = cb.dequantize(cb.quantize(Wv, cbk_v), cbk_v)
    vhat = rhat_v[:, None] * (QWv @ Pv)        # original-domain value recon (n_db, d)
    kl_acc = err_acc = denom = 0.0
    for j in range(n_q):
        Py = Qy[j] @ Pk.T
        s_true = Xk @ Qy[j]
        s_est = rhat_k * (QWk @ Py)
        p = np.exp(s_true - s_true.max()); p /= p.sum()
        phat = np.exp(s_est - s_est.max()); phat /= phat.sum()
        kl_acc += float(np.sum(p * np.log(p / np.clip(phat, 1e-12, None))))
        out_true = p @ Xv                    # (d,)
        out_hat = phat @ vhat if est_p else p @ vhat
        err_acc += float(np.sum((out_true - out_hat) ** 2))
        denom += float(np.sum(out_true ** 2))
    n = float(n_q)
    return {
        "rel_out_err": err_acc / max(denom, 1e-12),
        "kl": kl_acc / n,
        "bits": float((b_k + b_v) * d + 2 * NORM_BITS),
        "b_k": b_k, "b_v": b_v, "est_p": est_p,
    }


def attention_context_scale(d: int = 64, b: int = 2,
                            n_db_list: tuple = (128, 512, 1500, 4000, 8000),
                            n_q: int = 40, seed: int = SEED,
                            rot: str = "haar") -> list[dict]:
    """Attention-objective fidelity vs context length (exp 16).

    KV compression exists for LONG context, but every prior probe used one
    key-set size (n_db=1500). This sweeps the served context length with the
    SAME rotation (rot_seed), the SAME queries, and the same pipeline, so any
    change in softmax-KL / recall@5 / p95-logit-ratio / per-key score-noise is
    purely a context-length effect. Answers: does attention fidelity degrade
    as the cache grows (extreme-value logits, more misranking exposure), or
    is it context-stable?

    Returns list of dicts {n_db, kl, recall5, p95_logit_ratio, score_noise}.
    Deterministic given seed.
    """
    rows = []
    for n_db in n_db_list:
        m = attention_metrics(d, b, n_db=n_db, n_q=n_q, seed=seed, rot=rot)
        rows.append({"n_db": n_db, "kl": m["kl"], "recall5": m["recall5"],
                     "p95_logit_ratio": m["p95_logit_ratio"],
                     "score_noise": m["score_noise"]})
    return rows


def attention_protection(d: int = 64, n_db: int = 1500, n_q: int = 40,
                         seed: int = SEED, frac: float = 0.01,
                         prot_b: int = 3, base_b: int = 2,
                         mode: str = "topnorm") -> dict:
    """Per-key bit-reallocation on the attention objective (exp 13).

    Data-oblivious protected pool: the top `frac` of keys by STORED norm
    (rhat, available at encode time for free) get `prot_b`-bit direction
    codes, the rest get `base_b`. Avg true bits/token = mean(b_i)*d + 6-bit
    norm. `mode="random"` protects a random `frac` instead (control);
    `mode="topnorm"` protects the highest-norm keys.

    Returns dict with bits, kl (softmax-KL vs full precision), recall5,
    and the config. Deterministic given seed.
    """
    rng = np.random.default_rng(seed)
    P = pr.random_rotation(d, rng)
    X, U, R = realistic_dataset(d, n_db, seed ^ 0xA77)
    Qy, _, _ = realistic_dataset(d, n_q, seed ^ 0xB77)
    rhat = quantize_log_norm(R, NORM_BITS)
    k = max(1, int(round(frac * n_db)))
    prot = np.zeros(n_db, dtype=bool)
    if mode == "topnorm":
        order = np.argsort(rhat)[::-1]
        prot[order[:k]] = True
    else:
        perm = rng.permutation(n_db)
        prot[perm[:k]] = True
    bmap = np.where(prot, prot_b, base_b)
    w = U @ P.T
    QW = np.empty_like(w)
    for b in np.unique(bmap):
        m = bmap == b
        cbk = cb.beta_lloyd_max(d, int(b))
        QW[m] = cb.dequantize(cb.quantize(w[m], cbk), cbk)
    bits = float(np.mean(bmap) * d + NORM_BITS)
    kl_acc = r5 = 0.0
    for j in range(n_q):
        y = Qy[j]
        s_true = X @ y
        s_est = rhat * (QW @ (y @ P.T))
        p = np.exp(s_true - s_true.max()); p /= p.sum()
        q = np.exp(s_est - s_est.max()); q /= q.sum()
        kl_acc += float(np.sum(p * np.log(p / np.clip(q, 1e-12, None))))
        r5 += len(set(np.argsort(s_true)[-5:]) & set(np.argsort(s_est)[-5:])) / 5.0
    return {"bits": bits, "kl": kl_acc / n_q, "recall5": r5 / n_q,
            "frac": frac, "prot_b": prot_b, "base_b": base_b, "mode": mode}


def bias_raw_grid(nrot: int = 250, seed: int = SEED) -> float:
    """Mean |inner-product bias ratio - 1.0| over the grid (d, b) configs.

    The representative attention-logit fidelity metric: how far the estimator
    is from unbiased for the rates TurboQuant actually serves. THEORY-noise
    floor is the finite-d debias residual ~0.02-0.04.
    """
    vals = []
    for (d, b) in GRID_BIAS_CFGS:
        ratio = estimator_ratio(d, b, nrot, seed ^ (d * 131 + b))
        vals.append(abs(ratio - 1.0))
    return float(np.mean(vals))


def _product_estimates(d: int, m: int, n_trials: int, seed: int) -> np.ndarray:
    """Independent per-trial estimates of <y,x> by TurboQuant_prod (Alg 2):
    (b-1)=1-bit MSE base + m-bit QJL residual sketch. Matches the reference
    implementation in run_rotations.py (residual scaled by its own norm).
    Returns estimates."""
    x = pr.fixed_vector(d)
    y = pr.fixed_vector(d, kind="all_equal")
    rng = np.random.default_rng(seed)
    cbk1 = cb.beta_lloyd_max(d, 1)  # 1-bit MSE base
    ests = np.empty(n_trials)
    for i in range(n_trials):
        P = pr.random_rotation(d, rng)
        yx = P @ x
        idx = cb.quantize(yx, cbk1)
        base = P.T @ cb.dequantize(idx, cbk1)
        r = x - base
        gamma = np.linalg.norm(r)
        S = rng.standard_normal((m, d))
        ests[i] = (np.dot(y, base)
                   + gamma * np.sqrt(np.pi / 2) / m * np.sum((S @ y) * np.sign(S @ r)))
    return ests


def product_var_ratio_b2(d: int = 64, m: int = 64, n_trials: int = N_VAR,
                         seed: int = SEED) -> float:
    """Product-estimator variance / Theorem-2 bound.

    Bound: Dprod <= pi/(2d) * ||y||^2 * Dmse(b-1), with Dmse(1) measured.
    Ratio uses measured base MSE so it is fair at any d.
    """
    x = pr.fixed_vector(d)
    y = pr.fixed_vector(d, kind="all_equal")
    ests = _product_estimates(d, m, n_trials, seed)
    var = float(np.var(ests))
    cbk1 = cb.beta_lloyd_max(d, 1)
    rng = np.random.default_rng(seed)
    base_mse = 0.0
    nr = 300
    for _ in range(nr):
        P = pr.random_rotation(d, rng)
        yy = P @ x
        idx = cb.quantize(yy, cbk1)
        base_mse += float(np.sum((yy - cb.dequantize(idx, cbk1)) ** 2))
    base_mse /= nr
    bound = (np.pi / 2 / d) * float(np.dot(y, y)) * base_mse
    return float(var / bound)


# --------------------------------------------------------------------------- #
# Compute lightness: vectorized FWHT application time
# --------------------------------------------------------------------------- #
def fwht_apply_us(d: int = 128, reps: int = 40, seed: int = SEED) -> float:
    """Best-of-N wall-clock (µs) to apply the 3-round structured rotation."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(d)
    hdhdh = rot.hadamard_sign_flip(d, rng, rounds=3)
    best = float("inf")
    hdhdh(x)  # warmup
    for _ in range(reps):
        t0 = time.perf_counter()
        hdhdh(x)
        best = min(best, (time.perf_counter() - t0) * 1e6)
    return best


# --------------------------------------------------------------------------- #
# Self-check / invariants (used by .auto/checks.sh)
# --------------------------------------------------------------------------- #
def selfcheck() -> int:
    fail = []

    # 1. FWHT involutive property: fwht(fwht(x)) == d * x
    rng = np.random.default_rng(0)
    x = rng.standard_normal(128)
    d_ = x.shape[0]
    if not np.allclose(rot.fwht(rot.fwht(x.copy())), d_ * x, atol=1e-8):
        fail.append("fwht(fwht(x)) != d*x")

    # 1b. batch-safety (exp 18): fwht / hdhdh on (n, d) must equal per-row app
    Xb = rng.standard_normal((5, 128))
    Rb = rot.hadamard_sign_flip(128, rng, rounds=3)
    if not np.array_equal(Rb.forward(Xb),
                          np.array([Rb.forward(v) for v in Xb])):
        fail.append("hdhdh batch != per-row")
    if not np.allclose(Rb.inverse(Rb.forward(Xb)), Xb, atol=1e-9):
        fail.append("hdhdh batch roundtrip")
    if not np.array_equal(rot.fwht(Xb),
                          np.array([rot.fwht(v) for v in Xb])):
        fail.append("fwht batch != per-row")

    # 2. Structured rotation is orthogonal: <T x, T z> == <x, z>
    z = rng.standard_normal(128)
    hdhdh = rot.hadamard_sign_flip(128, rng, rounds=3)
    if not np.allclose(np.dot(hdhdh(x), hdhdh(z)), np.dot(x, z), atol=1e-8):
        fail.append("hdhdh not orthogonal")

    # 3. Benchmark metrics finite and in plausible ranges
    res = run(print_report=False)
    if not all(np.isfinite(v) for v in res["metrics"].values()):
        fail.append("non-finite metric")
    if res["metrics"]["tq_mse"] > 1.0:
        fail.append("tq_mse implausibly large")
    if res["metrics"]["tq_bias_raw_b1"] > 0.5:
        fail.append("tq_bias_raw_b1 implausibly large (debias broken?)")
    if res["metrics"]["tq_bias_raw"] > 0.5:
        fail.append("tq_bias_raw (grid) implausibly large (debias broken?)")
    if abs(res["metrics"]["tq_bytes_b2"] - (2 * 64 + NORM_BITS)) > 1e-6:
        fail.append(f"tq_bytes_b2 != {2 * 64 + NORM_BITS}"
                    f" (got {res['metrics']['tq_bytes_b2']})")
    if res["metrics"]["fwht_us"] < 1.0:
        fail.append("fwht_us implausibly small (under 1us)")

    if fail:
        print("SELFCHECK FAIL:", *fail, sep="\n  ")
        return 1
    print("SELFCHECK OK")
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(print_report: bool = True) -> dict:
    metrics = {}
    grid = grid_mse()
    metrics["tq_mse"] = grid["mean"]
    metrics["tq_bytes_b2"] = true_bits_b2()
    metrics["tq_bias_b1"] = bias_err_b1()
    metrics["tq_bias_raw_b1"] = bias_raw_err_b1()
    metrics["tq_bias_raw"] = bias_raw_grid()
    metrics["tq_var_b2"] = product_var_ratio_b2()
    metrics["tq_attn_kl_b2"] = attn_kl_b2()
    metrics["tq_attn_kl_b2_p95"] = attn_kl_b2_p95()
    metrics["fwht_us"] = fwht_apply_us()

    if print_report:
        print("OrbitQuant benchmark_tq  seed=%d" % SEED)
        print("  grid per-vector MSE at fixed honest rates (6-bit log norm header):")
        for (d, b), v in sorted(grid["per"].items()):
            print(f"    d={d:<3} b={b}  mse={v:.5f}")
        print(f"  tq_mse      (mean)        = {metrics['tq_mse']:.5f}   (lower better)")
        print(f"  tq_bytes_b2 (bits/token)  = {metrics['tq_bytes_b2']:.3f}   (lower better)")
        print(f"  tq_bias_b1  |ratio-2/pi|  = {metrics['tq_bias_b1']:.5f}   (lower better)")
        print(f"  tq_bias_raw_b1 |r-1| (uw) = {metrics['tq_bias_raw_b1']:.5f}   (lower better; debias={DEFAULT_DEBIAS})")
        print(f"  tq_bias_raw  grid|r-1|    = {metrics['tq_bias_raw']:.5f}   (lower better; representative)")
        print(f"  tq_var_b2   var/bound     = {metrics['tq_var_b2']:.5f}   (lower better)")
        print(f"  tq_attn_kl_b2 softmax-KLn = {metrics['tq_attn_kl_b2']:.5f}   (lower better; r/KL over realistic key set)")
        print(f"  tq_attn_kl_b2_p95 tail KL = {metrics['tq_attn_kl_b2_p95']:.5f}   (lower better; p95 of per-query KL, exp 17)")
        print(f"  fwht_us     d=128 best-of = {metrics['fwht_us']:.1f}   (lower better)")

    return {"grid": grid, "metrics": metrics}


def attention_kl_tail(d: int = 64, b: int = 2, n_q: int = 120,
                      n_db: int = 1500, seed: int = SEED,
                      frac: float = 0.0, prot_b: int = 3,
                      base_b: int = 2, mode: str = "topnorm") -> dict:
    """Query-level tail structure of attention fidelity (exp 17).

    Mean softmax-KL hides risk: a few queries with sharp attention (high query
    norm -> low temperature) or unlucky key-noise alignments may be served far
    worse than the mean suggests. Returns the per-query KL distribution
    (mean/p50/p90/p95/p99), the correlation of per-query KL with query norm,
    and the same stats for the data-oblivious protection variant (top-frac
    keys by stored norm at prot_b bits, rest at base_b, exp 13) so the tail
    benefit of protection can be separated from its (small) mean benefit.

    Deterministic given seed.
    """
    from . import codebooks as cbkmod
    rng = np.random.default_rng(seed)
    P = pr.random_rotation(d, rng)
    X, U, R = realistic_dataset(d, n_db, seed ^ 0xA77)
    Qy, _, Rq = realistic_dataset(d, n_q, seed ^ 0xB77)
    rhat = quantize_log_norm(R, NORM_BITS)

    def per_query_kl(bmap):
        w = U @ P.T
        QW = np.empty_like(w)
        for bb in np.unique(bmap):
            m = bmap == bb
            ck = cbkmod.beta_lloyd_max(d, int(bb))
            QW[m] = cbkmod.dequantize(cbkmod.quantize(w[m], ck), ck)
        kls = np.empty(n_q)
        for j in range(n_q):
            s_true = X @ Qy[j]
            s_est = rhat * (QW @ (Qy[j] @ P.T))
            p = np.exp(s_true - s_true.max()); p /= p.sum()
            q = np.exp(s_est - s_est.max()); q /= q.sum()
            kls[j] = float(np.sum(p * np.log(p / np.clip(q, 1e-12, None))))
        return kls

    bmap_all = np.full(n_db, b, dtype=np.int64)
    kls = per_query_kl(bmap_all)
    stats = lambda v: (float(np.mean(v)), float(np.median(v)),
                       float(np.percentile(v, 90)), float(np.percentile(v, 95)),
                       float(np.percentile(v, 99)))
    res = {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0,
           "qnorm_corr": 0.0}
    m, p50, p90, p95, p99 = stats(kls)
    res.update(mean=m, p50=p50, p90=p90, p95=p95, p99=p99)
    res["qnorm_corr"] = float(np.corrcoef(kls, Rq)[0, 1])
    if frac > 0.0:
        k = max(1, int(round(frac * n_db)))
        prot = np.zeros(n_db, dtype=bool)
        if mode == "topnorm":
            prot[np.argsort(rhat)[::-1][:k]] = True
        else:
            prot[rng.permutation(n_db)[:k]] = True
        bmap_p = np.where(prot, prot_b, base_b)
        kp = per_query_kl(bmap_p)
        mp, p50p, p90p, p95p, p99p = stats(kp)
        res.update(prot_mean=mp, prot_p50=p50p, prot_p90=p90p,
                   prot_p95=p95p, prot_p99=p99p, prot_frac=frac,
                   prot_b=prot_b, base_b=base_b, mode=mode)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfcheck", action="store_true", help="run invariants")
    args = ap.parse_args(argv)
    if args.selfcheck:
        return selfcheck()
    res = run()
    print("METRIC tq_mse=%.6f" % res["metrics"]["tq_mse"])
    print("METRIC tq_bytes_b2=%.3f" % res["metrics"]["tq_bytes_b2"])
    print("METRIC tq_bias_b1=%.6f" % res["metrics"]["tq_bias_b1"])
    print("METRIC tq_bias_raw_b1=%.6f" % res["metrics"]["tq_bias_raw_b1"])
    print("METRIC tq_bias_raw=%.6f" % res["metrics"]["tq_bias_raw"])
    print("METRIC tq_var_b2=%.6f" % res["metrics"]["tq_var_b2"])
    print("METRIC tq_attn_kl_b2=%.6f" % res["metrics"]["tq_attn_kl_b2"])
    print("METRIC tq_attn_kl_b2_p95=%.6f" % res["metrics"]["tq_attn_kl_b2_p95"])
    print("METRIC fwht_us=%.2f" % res["metrics"]["fwht_us"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
