"""OrbitQuant statistical baseline runner.

Verifies TurboQuant's core theoretical claims on random unit vectors, with no
model and no GPU. Pure NumPy. Deterministic given the seed.

Usage:
    python -m baseline.run_baseline [--seed N] [--d 64] [--n 20000]

Exits nonzero if any check fails.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import codebooks as cb
from . import checks


def _fmt(v) -> str:
    if isinstance(v, tuple):
        return "(" + ", ".join(f"{x:.4f}" for x in v) + ")"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=64, help="head dimension")
    ap.add_argument("--nrot", type=int, default=2000, help="independent rotations/trials")
    ap.add_argument("--m", type=int, default=64, help="QJL sketch rows")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    d, nrot, m = args.d, args.nrot, args.m

    print(f"OrbitQuant statistical baseline")
    print(f"  d={d}  nrot={nrot}  m={m}  seed={args.seed}")
    print()

    # --- Codebook comparison (P0.3): exact-Beta vs Gaussian closed-form ---
    print("Codebooks (positive-side centroids, d=%d):" % d)
    for b in (1, 2):
        beta = cb.beta_lloyd_max(d, b)
        gauss = cb.gaussian_closed_form(d, b)
        print(f"  b={b}  beta={np.round(beta,4)}  gauss={np.round(gauss,4)}")
    print()

    # --- Run the six statistical checks ---
    results = []
    for name, fn in checks.ALL_CHECKS.items():
        if name in ("mse_distortion", "inner_product_bias", "near_optimality"):
            measured, expected, tol, passed = fn(d, 1, nrot, rng)
        elif name == "qjl":
            measured, expected, tol, passed = fn(d, m, nrot, rng)
        else:
            measured, expected, tol, passed = fn(d, nrot, rng)
        results.append((name, measured, expected, passed))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name:24s} measured={_fmt(measured):30s} "
              f"expected={_fmt(expected):30s}")

    # --- MSE distortion at multiple bit-widths (the paper's headline numbers) ---
    print()
    print("MSE distortion vs bit-width (per-vector, d=%d):" % d)
    for b in (1, 2, 3, 4):
        _, mse, ref, _ = checks.check_mse_distortion(d, b, nrot, rng)
        print(f"  b={b}  measured={mse:.4f}  paper_asymptotic={ref:.4f}")

    n_fail = sum(1 for _, _, _, p in results if not p)
    print()
    if n_fail:
        print(f"FAILED: {n_fail} check(s) failed")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
