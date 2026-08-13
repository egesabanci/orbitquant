"""P0.5 — True byte accounting and deployability score runner.

Usage:
    python3 -m baseline.run_p05 --d 64 --n 8192
"""
from __future__ import annotations

import argparse
import sys

from . import p05_accounting as p05


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=p05.__doc__)
    ap.add_argument("--d", type=int, default=64, help="head dimension")
    ap.add_argument("--n", type=int, default=8192, help="cached tokens")
    args = ap.parse_args(argv)

    print(f"P0.5 true byte accounting and deployability  d={args.d} n_tokens={args.n}")
    print("  counts ALL metadata (payload, padding, side info, codebook IDs,")
    print("  protected pools, predictor weights, layout) -- per P0.5/proposals.md")
    print()

    rows = p05.run_accounting(args.d, args.n)

    hdr = (f"{'representation':<34}{'nom':>5}{'payload':>8}{'pad':>6}"
           f"{'side':>6}{'prot':>6}{'pred':>7}{'TOTAL b/t':>10}{'b/coord':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        a = r["acc"]
        print(f"{r['rep'].name:<34}{a['nominal_bits_per_coord']:>5.2f}"
              f"{a['payload_bits']:>8.0f}{a['packing_padding_bits']:>6.1f}"
              f"{a['side_info_bits']:>6.1f}"
              f"{a['protected_meta_bits'] + a['protected_extra_bits']:>6.1f}"
              f"{a['predictor_amortized_bits']:>7.3f}"
              f"{a['total_bits_per_token']:>10.1f}"
              f"{a['true_bits_per_coord_equiv']:>8.3f}")
    print()

    print("Deployability (structural=verified from codec, fused/dequant=ASSUMED")
    print("until P2.1 GPU kernels are run):")
    print(f"{'representation':<34}{'verif':>6}{'inclAsm':>8}   paged reg fused batch no_dequant")
    for r in rows:
        dep = r["dep"]
        s = dep["structural"]
        a = dep["assumed_gpu"]
        flags = "".join(
            ("?" if v else ".") if k in a else ("Y" if v else ".")
            for k, v in {**s, **a}.items()
        )
        print(f"{r['rep'].name:<34}{dep['verified_score']:>6.2f}"
              f"{dep['score_incl_assumed']:>8.2f}   {flags}")
    print("  '?' = assumed (GPU kernel not run); Y = verified structural")
    print()

    # headline comparison: nominal vs true at the same nominal bits
    print("Nominal vs true (the P0.5 point):")
    for r in rows:
        a = r["acc"]
        ratio = a["true_bits_per_coord_equiv"] / a["nominal_bits_per_coord"]
        print(f"  {r['rep'].name:<34} nominal {a['nominal_bits_per_coord']:.2f} "
              f"-> true {a['true_bits_per_coord_equiv']:.3f} "
              f"({ratio:.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
