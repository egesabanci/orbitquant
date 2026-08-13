"""P1.1/P1.7/P1.8 — Mixed mode, analytical debiasing, norm-preserving reconstruction.

Benchmarks the P1.1 mode-selection policy over the four candidate modes at the
SAME total bit budget:

- ``mse`` (TurboQuant_mse, b bits): plain MSE quantizer; biased for inner
  products (~2/pi at b=1).
- ``debiased`` (b bits): MSE quantizer + multiplicative bias correction (P1.7).
- ``norm_preserving`` (b bits): MSE quantizer + renormalize to original norm
  (P1.8).
- ``product`` (TurboQuant_prod, b total bits): (b-1)-bit MSE base + 1-bit QJL
  residual sketch (Algorithm 2, matched total bits via P2.4/P2.2 machinery).

For each mode we report the inner-product bias ratio (1.0 = unbiased) and
per-vector MSE, then the mixed-mode policy selection and the measured aggregate
of the selected mode per trial.

Usage:
    python3 -m baseline.run_corrections --d 64 --nrot 2000 --b 1
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import codebooks as cb
from . import corrections as corr
from . import protocol as pr


def eval_mode(mode: str, d: int, n_rot: int, b: int, correction: float,
              rng: np.random.Generator) -> tuple:
    """Run one mode over n_rot trials; return (bias_ratio, mse, per-trial bias)."""
    x = pr.fixed_vector(d)
    y = pr.fixed_vector(d, kind="all_equal")
    cbk = cb.beta_lloyd_max(d, b)          # b-bit base for mse/debiased/norm
    # b-1-bit base for product; at b=1 the base is 0 bits -> pure QJL (paper
    # Algorithm 2 with b-1=0).
    cbk_base = cb.beta_lloyd_max(d, b - 1) if b > 1 else None
    true = float(np.dot(y, x))
    bias_ests = np.empty(n_rot)
    mse_ests = np.empty(n_rot)
    for i in range(n_rot):
        P = pr.random_rotation(d, rng)
        yx = P @ x
        if mode == "mse":
            xhat = P.T @ cb.dequantize(cb.quantize(yx, cbk), cbk)
        elif mode == "debiased":
            xhat = corr.debiased_quantize(x, cbk, P, correction)
        elif mode == "norm_preserving":
            xhat = corr.norm_preserving_quantize(x, cbk, P)
        else:  # product: (b-1)-bit MSE base + 1-bit QJL residual = b total bits
            if b > 1:
                idx = cb.quantize(yx, cbk_base)
                xhat_base = P.T @ cb.dequantize(idx, cbk_base)
            else:
                xhat_base = np.zeros(d)  # b=1: pure QJL, no MSE base
            r = x - xhat_base
            gamma = np.linalg.norm(r)
            S = rng.standard_normal((d, d))
            xhat = xhat_base + gamma * np.sqrt(np.pi / 2) / d * (
                S.T @ np.sign(S @ r)
            )
        bias_ests[i] = np.dot(y, xhat)
        mse_ests[i] = np.sum((x - xhat) ** 2)
    bias_ratio = float(np.mean(bias_ests) / true)
    mse = float(np.mean(mse_ests))
    return bias_ratio, mse, bias_ests


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d, n_rot, b = args.d, args.nrot, args.b
    rng = np.random.default_rng(args.seed)
    x = pr.fixed_vector(d)
    y = pr.fixed_vector(d, kind="all_equal")
    true = float(np.dot(y, x))

    # bias ratio of the plain MSE quantizer (P1.7 baseline)
    bias_mse = corr.bias_correction_factor(d, b, n_rot, args.seed)
    correction = 1.0 / bias_mse

    print(f"P1.1 mixed-mode benchmark  d={d} nrot={n_rot} b={b} seed={args.seed}")
    print(f"  plain MSE bias ratio: {bias_mse:.4f} (2/pi={2/np.pi:.4f})")
    print(f"  debias correction: {correction:.4f}")
    print()

    modes = ["mse", "debiased", "norm_preserving", "product"]
    print(f"{'mode':<16}{'bias_ratio':>12}{'mse':>10}{'unbiased?':>10}{'bits':>8}")
    print("-" * 56)
    results = {}
    per_trial = {}
    for mode in modes:
        bias_ratio, mse, bias_ests = eval_mode(mode, d, n_rot, b, correction, rng)
        results[mode] = {"bias_ratio": bias_ratio, "mse": mse,
                         "unbiased": abs(bias_ratio - 1.0) < 0.05}
        per_trial[mode] = bias_ests
        bits = b * d if mode != "product" else (b - 1) * d + d
        print(f"{mode:<16}{bias_ratio:>12.4f}{mse:>10.4f}"
              f"{str(results[mode]['unbiased']):>10}{bits:>8}")

    # P1.1 mixed-mode policy: choose per trial the mode with the best objective.
    # Policy A (min_mse_unbiased): among unbiased modes, lowest MSE; product
    #   mode is the guaranteed-unbiased fallback.
    # Policy B (min_bias_within_1.5x_mse): lowest bias for MSE within 1.5x of
    #   plain MSE.
    print()
    print("P1.1 mixed-mode policy selection:")
    # Policy A: prefer unbiased, lowest MSE among unbiased modes
    unbiased_modes = [m for m, r in results.items() if r["unbiased"]]
    if unbiased_modes:
        best = min(unbiased_modes, key=lambda m: results[m]["mse"])
        print(f"  min_mse_unbiased: {best} (mse={results[best]['mse']:.4f}, "
              f"bias={results[best]['bias_ratio']:.4f})")
    else:
        print("  min_mse_unbiased: none unbiased; product mode would be required")
    # Policy B: lowest bias for MSE within 1.5x of plain MSE
    mse_mse = results["mse"]["mse"]
    affordable = [m for m, r in results.items() if r["mse"] <= 1.5 * mse_mse]
    if affordable:
        best = min(affordable, key=lambda m: abs(results[m]["bias_ratio"] - 1.0))
        print(f"  min_bias_within_1.5x_mse: {best} "
              f"(bias={results[best]['bias_ratio']:.4f}, mse={results[best]['mse']:.4f})")
    else:
        print("  min_bias_within_1.5x_mse: none affordable")

    # Per-trial aggregate: select the min-mse-unbiased mode per trial and
    # measure the resulting aggregate bias and MSE.
    pool = unbiased_modes or ["mse"]
    sel = np.empty(n_rot)
    for i in range(n_rot):
        best_m = min(pool, key=lambda m: (results[m]["mse"], i))
        sel[i] = per_trial[best_m][i]
    sel_bias = float(np.mean(sel) / true)
    print(f"  aggregate (per-trial min-mse-unbiased over {pool}): "
          f"bias_ratio={sel_bias:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
