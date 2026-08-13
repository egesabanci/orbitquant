"""P0.1 — Rotation validation harness.

Compares rotation transforms on the same statistical checks used for the
baseline: for a FIXED adversarial vector x averaged over independent rotations,
does each rotation produce Beta-distributed, near-independent coordinates, and
does it preserve MSE / inner-product-bias / near-optimality?

This is the gating experiment: structured rotations (P0.2) may work well in
practice but do NOT inherit Haar's exact guarantees, so we measure whether they
preserve the assumptions TurboQuant relies on.

Usage:
    python -m baseline.run_rotations --d 64 --nrot 2000
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import codebooks as cb
from . import protocol as pr
from . import rotations as rot


def eval_rotation(name: str, d: int, n_rot: int, rng: np.random.Generator, b: int = 1):
    """Run the statistical checks for one rotation transform."""
    x = pr.fixed_vector(d)
    y = pr.fixed_vector(d, kind="all_equal")
    cbk = cb.beta_lloyd_max(d, b)

    samples = np.empty(n_rot * d)
    mse = 0.0
    bias_ests = np.empty(n_rot)
    prod_ests = np.empty(n_rot)
    t0 = time.perf_counter()
    for i in range(n_rot):
        P = rot.rotation_from_name(name, d, rng)
        yx = P @ x
        samples[i * d : (i + 1) * d] = yx
        idx = cb.quantize(yx, cbk)
        yhat = cb.dequantize(idx, cbk)
        mse += float(np.sum((yx - yhat) ** 2))
        xhat = P.T @ yhat
        bias_ests[i] = np.dot(y, xhat)
        # Full TurboQuant-product estimator (Algorithm 2): MSE base score plus
        # the QJL residual, scaled by ||r||. This is what Lemma 4 bounds.
        r = x - xhat
        gamma = np.linalg.norm(r)
        S = rng.standard_normal((d, d))
        prod_ests[i] = np.dot(y, xhat) + gamma * np.sqrt(np.pi / 2) / d * np.sum(
            (S @ y) * np.sign(S @ r)
        )
    elapsed = time.perf_counter() - t0

    ks = pr.beta_ks(d, samples)
    mse /= n_rot
    true = float(np.dot(y, x))
    bias_ratio = float(np.mean(bias_ests) / true)
    floor = 1.0 / (4**b)
    gap = mse / floor

    # covariance / higher-order dependence: distinct rotated coords ~independent
    Y = samples.reshape(n_rot, d)
    corr = np.corrcoef(Y, rowvar=False)
    off = corr - np.eye(d)
    rms_off = float(np.sqrt(np.mean(off**2)))
    c1, c2 = Y[:, 0], Y[:, 1]
    lhs = np.mean(c1**2 * c2**2)
    rhs = np.mean(c1**2) * np.mean(c2**2)
    factor_gap = float(abs(lhs - rhs) / (rhs + 1e-12))

    # Full product estimator: bias and variance vs the Lemma 4/Thm 2 bound
    prod_bias = float(np.mean(prod_ests) - true)
    prod_var = float(np.var(prod_ests))
    # Thm 2 bound: (sqrt(3)pi/2)*||y||^2/d * 1/4^b
    prod_bound = (np.sqrt(3) * np.pi / 2) * np.dot(y, y) / d / (4**b)

    return {
        "rotation": name,
        "beta_ks": ks,
        "rms_corr": rms_off,
        "factor_gap": factor_gap,
        "mse": mse,
        "bias_ratio": bias_ratio,
        "prod_bias": prod_bias,
        "prod_var": prod_var,
        "prod_bound": prod_bound,
        "near_optimality_gap": gap,
        "runtime_s": elapsed,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--rotations",
        nargs="*",
        default=["haar", "hadamard", "hd", "hdhd", "hdhdh", "perm"],
    )
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    d, n_rot = args.d, args.nrot

    print(f"P0.1 Rotation validation harness  d={d} nrot={n_rot} seed={args.seed}")
    print(f"{'rotation':<10}{'beta_ks':>9}{'rms_corr':>10}{'fact_gap':>9}"
          f"{'mse':>9}{'bias':>9}{'p_bias':>9}{'p_var':>9}{'opt_gap':>9}{'rt_s':>7}")
    print("-" * 90)
    for name in args.rotations:
        r = eval_rotation(name, d, n_rot, rng)
        print(f"{r['rotation']:<10}{r['beta_ks']:>9.4f}{r['rms_corr']:>10.4f}"
              f"{r['factor_gap']:>9.4f}{r['mse']:>9.4f}{r['bias_ratio']:>9.4f}"
              f"{r['prod_bias']:>9.4f}{r['prod_var']:>9.4f}"
              f"{r['near_optimality_gap']:>9.3f}{r['runtime_s']:>7.3f}")

    # reference values for the ideal Haar rotation
    print("-" * 90)
    print("Reference (Haar, d=64): beta_ks~0.002, rms_corr~0.02, mse~0.36, "
          "bias~0.637, prod_bias~0, prod_var<bound, opt_gap~1.44")
    return 0


if __name__ == "__main__":
    sys.exit(main())
