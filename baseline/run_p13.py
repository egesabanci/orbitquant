"""P1.3 — Quantized norm/radius storage benchmark runner.

Usage:
    python3 -m baseline.run_p13 --d 64 --nrot 2000
"""
from __future__ import annotations

import argparse
import sys

from . import p13_norm as p13


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=p13.__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    step, rel = p13.format_error(bits=args.bits)
    print(f"P1.3 quantized norm/radius storage  d={args.d} nrot={args.nrot} "
          f"bits={args.bits} seed={args.seed}")
    print(f"  log2-format step={step:.4f}  max rel error={rel:.4%} "
          f"(dynamic range 2^{{-4}}..2^{{4}})")
    print()

    res = p13.measure(
        d=args.d, n_rot=args.nrot, b=1, bits=args.bits, seed=args.seed
    )
    print("  norm relative error (full precision):  "
          f"{res['rel_err_unit']:.6f}")
    print("  norm relative error (8-bit log, unit): "
          f"{res['rel_err_rec']:.6f}")
    print("  inner-product bias (full precision):   "
          f"{res['beta_fp']:.4f}")
    print("  inner-product bias (quantized unit):   "
          f"{res['beta_q']:.4f}")
    print("  inner-product bias (quantized per-vec):"
          f"{res['beta_qdist']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
