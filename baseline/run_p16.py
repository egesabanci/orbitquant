"""P1.6 — Adaptive bit allocation runner.

Measures per-coordinate variance of the rotated fixed vector over a
calibration pass, water-fills a fixed total bit budget across coordinates by
marginal distortion gain, and compares uniform vs adaptive allocation at the
same budget (protocol: fixed e1, independent Haar rotations per trial). Also
reports the per-coordinate bit distribution and a synthetic outlier-variance
mechanism check.

Usage:
    python3 -m baseline.run_p16 --d 64 --nrot 2000 [--bits 2] [--ncalib 1000]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np

from . import codebooks as cb
from . import p16_adaptive as p16
from . import protocol as pr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=p16.__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--bits", type=int, default=2,
                    help="average per-coordinate bits (budget = bits*d)")
    ap.add_argument("--ncalib", type=int, default=1000,
                    help="rotations in the variance calibration pass")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d, n_rot, bits = args.d, args.nrot, args.bits
    if not (p16.B_MIN_BITS <= bits <= p16.B_MAX_BITS):
        ap.error(f"--bits must be in [{p16.B_MIN_BITS}, {p16.B_MAX_BITS}]")
    budget = bits * d
    rng = np.random.default_rng(args.seed)

    # rate-distortion curve of the exact codebooks (per unit variance)
    drel = p16.distortion_curve(d)
    print(f"P1.6 adaptive bit allocation  d={d} nrot={n_rot} bits={bits} "
          f"budget={budget} ncalib={args.ncalib} seed={args.seed}")
    print("  D_rel(b) (per-unit-variance distortion, Beta Lloyd-Max codebooks): "
          + ", ".join(f"b={i}: {v:.4f}" for i, v in enumerate(drel, start=p16.B_MIN_BITS)))
    print()

    # calibration: per-coordinate variance over independent rotations
    var = p16.measured_coordinate_variance(d, args.ncalib, rng)
    print("  measured per-coordinate variance (e1, "
          f"{args.ncalib} rotations): mean={var.mean():.6f} "
          f"min={var.min():.6f} max={var.max():.6f} "
          f"ratio(max/min)={var.max() / var.min():.3f}")

    # allocations at the matched budget
    alloc = p16.allocate_bits(var, drel, budget)
    b_u = budget // d
    print(f"  uniform bits:  {str(np.full(d, b_u, dtype=int)).replace(chr(10), '')}")
    print(f"  adaptive bits: {str(alloc).replace(chr(10), '')}")
    print("  adaptive bit distribution: "
          + ", ".join(f"{k} bits x {v}" for k, v in sorted(Counter(alloc).items())))
    print()

    # matched-budget MSE comparison (independent Haar rotations per trial)
    codebooks = {b: cb.beta_lloyd_max(d, b) for b in np.unique(alloc)}
    codebooks.setdefault(b_u, cb.beta_lloyd_max(d, b_u))
    mse_uni = p16.uniform_mse(d, n_rot, b_u, codebooks[b_u], rng)
    mse_ada = p16.adaptive_mse(d, n_rot, alloc, codebooks, rng)
    gain = (mse_ada - mse_uni) / mse_uni
    print(f"  uniform MSE  (b={b_u} x {d}): {mse_uni:.6f}")
    print(f"  adaptive MSE (budget {budget}):   {mse_ada:.6f}")
    print(f"  relative gain (adaptive-uniform)/uniform: {gain:+.4%} "
          "(negative = adaptive better)")
    print()

    # mechanism check: allocator under genuine variance structure
    print("  synthetic outlier profile (mechanism check, no rotation): "
          f"n_outliers=4 var=10/d, budget={bits * d}")
    synth = p16.synthetic_outlier_eval(d, n_rot, drel, b_avg=bits, seed=args.seed)
    print("    adaptive bit distribution: "
          + ", ".join(f"{k} bits x {v}" for k, v in sorted(Counter(synth["allocation"]).items())))
    g = (synth["adaptive_mse"] - synth["uniform_mse"]) / synth["uniform_mse"]
    print(f"    uniform MSE  (b={synth['b_uniform']}): {synth['uniform_mse']:.6f}")
    print(f"    adaptive MSE (budget {synth['budget']}): {synth['adaptive_mse']:.6f}")
    print(f"    relative gain: {g:+.4%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
