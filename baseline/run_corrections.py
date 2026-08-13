"""P1.1/P1.7/P1.8 — Mixed mode, analytical debiasing, norm-preserving reconstruction.

Benchmarks the cheap post-dequantization corrections on the statistical checks:

- P1.7 ``debiased``: MSE quantizer + multiplicative bias correction.
- P1.8 ``norm_preserving``: MSE quantizer + renormalize to original norm.
- P1.1 ``mixed``: choose per-coordinate mode -- for each coordinate, use the
  MSE codebook, and apply the debiasing correction (a cheap alternative to
  residual QJL when it is sufficient).

For each mode we report the inner-product bias ratio (should be ~1.0 for
debiased/mixed, i.e. unbiased) and the per-vector MSE.

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
    cbk = cb.beta_lloyd_max(d, b)
    true = float(np.dot(y, x))

    # bias ratio of the plain MSE quantizer (P1.7 baseline)
    bias_mse = corr.bias_correction_factor(d, b, n_rot, args.seed)
    correction = 1.0 / bias_mse

    print(f"P1 corrections benchmark  d={d} nrot={n_rot} b={b} seed={args.seed}")
    print(f"  plain MSE bias ratio: {bias_mse:.4f} (2/pi={2/np.pi:.4f})")
    print(f"  debias correction: {correction:.4f}")
    print()

    modes = ["mse", "debiased", "norm_preserving"]
    print(f"{'mode':<16}{'bias_ratio':>12}{'mse':>10}{'unbiased?':>10}")
    print("-" * 48)
    for mode in modes:
        bias_ests = np.empty(n_rot)
        mse_ests = np.empty(n_rot)
        for i in range(n_rot):
            P = pr.random_rotation(d, rng)
            if mode == "mse":
                xhat = P.T @ cb.dequantize(cb.quantize(P @ x, cbk), cbk)
            elif mode == "debiased":
                xhat = corr.debiased_quantize(x, cbk, P, correction)
            else:
                xhat = corr.norm_preserving_quantize(x, cbk, P)
            bias_ests[i] = np.dot(y, xhat)
            mse_ests[i] = np.sum((x - xhat) ** 2)
        bias_ratio = float(np.mean(bias_ests) / true)
        mse = float(np.mean(mse_ests))
        unbiased = abs(bias_ratio - 1.0) < 0.05
        print(f"{mode:<16}{bias_ratio:>12.4f}{mse:>10.4f}{str(unbiased):>10}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
