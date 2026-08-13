"""P2.6 — Importance-weighted bit allocation runner.

Calibration pass estimates per-coordinate sensitivity; bits are then allocated
to important coordinates under a fixed total budget (greedy, D(b) =
sum_i w_i 2^-2b_i with w_i = y_i^2) and compared against uniform allocation of
the same budget. Protocol: fixed adversarial x (e1), independent sign-corrected
Haar rotations per trial.

Two honest readings (per P0.5 byte accounting, allocation metadata is PAID):

A. Fixed allocation from calibration: the pattern is computed ONCE from the
   calibration sensitivity and applied to every trial; nothing is signaled per
   vector -> matched budget with uniform, no side info. Under Haar rotations
   coordinates are exchangeable, so this reduces to (near-)uniform.

B. Per-vector adaptive allocation: the pattern is recomputed per trial from the
   realized importance w = y^2. The pattern must be signaled to the decoder at
   S = d*ceil(log2(bmax+1)) bits, CHARGED against the budget, so the honest
   comparison is: uniform(B_uniform bits) vs adaptive(B_quant + S bits) with
   B_quant + S == B_uniform.

Usage:
    python3 -m baseline.run_p26 --d 64 --nrot 2000 [--budget 128] [--bmax 4]
                                [--ncal 64] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import codebooks as cb
from . import p26_importance as imp
from . import protocol as pr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--budget", type=int, default=None,
                    help="total quantizer bit budget (default 2*d)")
    ap.add_argument("--bmax", type=int, default=4,
                    help="max bits per coordinate for the allocator")
    ap.add_argument("--ncal", type=int, default=64,
                    help="rotations in the calibration pass")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d, n_rot = args.d, args.nrot
    budget = args.budget if args.budget is not None else 2 * d
    rng = np.random.default_rng(args.seed)
    x = pr.fixed_vector(d)

    b_u = budget // d
    if b_u < 1:
        print(f"error: budget {budget} < d {d}: no uniform b>=1 comparison possible")
        return 2
    if budget > args.bmax * d:
        print(f"error: budget {budget} exceeds bmax*d = {args.bmax}*{d}")
        return 2

    codebooks = {b: cb.beta_lloyd_max(d, b) for b in range(1, args.bmax + 1)}
    S = d * int(np.ceil(np.log2(args.bmax + 1)))  # signaling cost of the pattern

    print(f"P2.6 importance-weighted bit allocation  d={d} nrot={n_rot} "
          f"budget={budget} bmax={args.bmax} ncal={args.ncal} seed={args.seed}")
    print(f"  x = e1 (fixed, adversarial), independent Haar rotations per trial")
    print(f"  signaling cost of a per-vector allocation pattern: S = {S} bits")

    # --- calibration pass: per-coordinate sensitivity over independent rotations ---
    sens = imp.calibrate_sensitivity(x, args.ncal, rng)
    print(f"  calibration ({args.ncal} rotations): mean sensitivity = "
          f"{sens.mean():.5f} (1/d = {1.0 / d:.5f}), max/min = "
          f"{sens.max() / sens.min():.2f}")
    print("    -> coordinates exchangeable under Haar; allocation driven by "
          "per-vector importance w_i = y_i^2")
    print()

    # --- uniform allocation at the same budget (b_u*d bits) ---
    mse_uniform = imp.uniform_mse(x, b_u, codebooks[b_u], n_rot, rng)
    print(f"uniform allocation     b={b_u}  budget={b_u * d:>4} bits  "
          f"mse = {mse_uniform:.5f}")
    print()

    # --- A. fixed calibration-derived allocation (no side info, matched budget) ---
    fixed_bits = imp.allocate_bits(sens, b_u * d, args.bmax)
    mse_fixed = imp.fixed_allocation_mse(x, fixed_bits, codebooks, n_rot, rng)
    print(f"A. fixed calibration mask (no per-vector side info):")
    print(f"   budget={b_u * d:>4} bits (same as uniform)  mse = {mse_fixed:.5f}")
    print(f"   gain vs uniform: {mse_uniform / mse_fixed:.3f}x "
          f"({'better' if mse_fixed < mse_uniform else 'worse'})")
    print()

    # --- B. per-vector adaptive allocation WITH signaling charged ---
    # Honest comparison: uniform(B_uniform) vs adaptive(B_quant + S), B_quant+S==B_uniform.
    b_quant = b_u * d - S
    if b_quant < d:
        print("B. per-vector adaptive: signaling S exceeds budget for any b>=1 "
              "per-coordinate; not comparable at this budget.")
    else:
        mse_adapt, counts = imp.allocation_stats(x, b_quant, args.bmax, codebooks,
                                                 n_rot, rng)
        print(f"B. per-vector adaptive allocation WITH signaling charged:")
        print(f"   quantizer bits = {b_quant} + signaling {S} = "
              f"{b_quant + S} total (== uniform {b_u * d})  mse = {mse_adapt:.5f}")
        print(f"   gain vs uniform: {mse_uniform / mse_adapt:.3f}x "
              f"({'better' if mse_adapt < mse_uniform else 'worse'})")
        print("   bit distribution (mean #coords of %d per width):" % d)
        line = []
        for b in range(args.bmax + 1):
            line.append(f"b={b}: {counts[b]:.1f}")
        print("    " + "   ".join(line))

    return 0


if __name__ == "__main__":
    sys.exit(main())
