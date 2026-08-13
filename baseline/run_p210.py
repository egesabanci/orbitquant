"""P2.10 — Predictive residual coding runner.

Builds a synthetic correlated layer-key chain (Markov, successive correlation
rho, started from the fixed adversarial e1, then a fixed isometry), trains a
compact linear predictor on the first n_cal layers (directly-quantized
previous reconstructions), and compares per-test-layer MSE of direct
quantization vs predictive residual coding (only the residual is quantized)
at b = 1, 2 with the protocol's exact Beta Lloyd-Max codebooks.

Usage:
    python3 -m baseline.run_p210 --d 64 --nrot 2000 [--rho 0.9] [--nlayers 12]
                                 [--ncal 8] [--rank both] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys

from . import p210_predictive as p210


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=p210.__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--rho", type=float, default=0.9,
                    help="layer-to-layer correlation of the synthetic chain")
    ap.add_argument("--nlayers", type=int, default=12)
    ap.add_argument("--ncal", type=int, default=8,
                    help="calibration layers (train split; test = ncal..end)")
    ap.add_argument("--rank", choices=["scalar", "rank1", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    ranks = ["scalar", "rank1"] if args.rank == "both" else [args.rank]
    res = p210.measure(
        d=args.d, n_rot=args.nrot, n_layers=args.nlayers, n_cal=args.ncal,
        rho=args.rho, ranks=ranks, b_values=(1, 2), seed=args.seed,
    )
    n_test = args.nlayers - args.ncal

    print(f"P2.10 predictive residual coding  d={args.d} nrot={args.nrot} "
          f"rho={args.rho} nlayers={args.nlayers} ncal={args.ncal} "
          f"seed={args.seed}")
    print(f"  correlated layer chain (Markov, rho={args.rho}, start e1) + "
          f"fixed isometry; test layers {args.ncal}..{args.nlayers - 1} "
          f"({n_test})")
    print()

    for b in res["b_values"]:
        r = res[b]
        d = r["direct_mse"]
        print(f"  b={b}  codebook 2**{b} Beta Lloyd-Max")
        print(f"    direct quantization MSE (current key alone):  {d:.6f}")
        for name in ranks:
            p = r[name]
            amort = p["storage_bits"] / (args.nlayers * args.d)
            print(f"    predictive ({name:>5}) MSE: {p['mse']:.6f}   "
                  f"= {p['mse'] / d:.3f}x direct   "
                  f"residual energy {p['gain']:.4f}   "
                  f"predictor {p['storage_bits']} b total "
                  f"({amort:.2f} b/coord amortized)")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
