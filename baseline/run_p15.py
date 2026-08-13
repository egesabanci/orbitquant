"""P1.5 — Protected-token pools (runner).

Benchmarks a pool of N unit vectors (tokens) where a tiny protected subset
(attention sinks / outliers, the first k tokens) is quantized at b_hi bits
while the rest use b_lo bits, vs uniform quantization of the whole pool at the
same total payload budget (quantized index bits only, no metadata). All
measured values; no pass/fail thresholds.

Usage:
    python3 -m baseline.run_p15 --d 64 --nrot 2000 --n 1000
    python3 -m baseline.run_p15 --d 64 --nrot 2000 --n 1000 \
        --pairs "1,3;2,4" --fracs "0.01,0.05"
"""
from __future__ import annotations

import argparse
import sys
import time

from . import p15_protected as p15


def parse_pairs(s: str) -> list[tuple[int, int]]:
    """Parse "b_lo,b_hi;b_lo,b_hi" into [(b_lo, b_hi), ...]."""
    pairs = []
    for part in s.split(";"):
        if not part.strip():
            continue
        lo, hi = part.split(",")
        pairs.append((int(lo), int(hi)))
    if not pairs:
        raise ValueError(f"no pairs parsed from {s!r}")
    return pairs


def parse_fracs(s: str) -> list[float]:
    fracs = [float(x) for x in s.split(",") if x.strip()]
    if not fracs:
        raise ValueError(f"no fractions parsed from {s!r}")
    return fracs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--n", type=int, default=1000, help="pool size (tokens)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fracs", type=str, default="0.01,0.05",
                    help="protected pool fractions (comma-separated)")
    ap.add_argument("--pairs", type=str, default="1,3;2,4",
                    help="(b_lo,b_hi) bit pairs (semicolon-separated)")
    args = ap.parse_args(argv)

    fracs = parse_fracs(args.fracs)
    pairs = parse_pairs(args.pairs)

    t0 = time.perf_counter()
    res = p15.benchmark(
        d=args.d, n=args.n, n_rot=args.nrot,
        seed=args.seed, fracs=tuple(fracs), pairs=tuple(pairs),
    )
    elapsed = time.perf_counter() - t0

    print(f"P1.5 protected-token pools  d={res['d']} n={res['n']} "
          f"nrot={res['n_rot']} seed={res['seed']}  ({elapsed:.1f}s)")
    print("pool: fixed N unit vectors (iid Gaussian directions, seed-fixed)")
    print("protected subset = first k tokens (attention sinks); payload bits "
          "only, no metadata")
    print()

    print("uniform reference (whole pool at one integer rate, measured):")
    for b, mse in res["uniform"].items():
        bits = res["d"] * res["n"] * b
        print(f"  rate {b} bit:  mse={mse:.6f}  payload={bits} bits")
    print()

    hdr = (f"{'frac':>6}{'b_lo':>5}{'b_hi':>5}{'k':>5}"
           f"{'avg_bits':>9}{'payload':>10}"
           f"{'pool_mse':>10}{'prot_mse':>10}{'reg_mse':>10}"
           f"{'uni_eq':>10}{'ratio':>8}")
    print(hdr)
    print("-" * len(hdr))
    for c in res["configs"]:
        print(
            f"{c['frac']:>6.2%}{c['b_lo']:>5}{c['b_hi']:>5}{c['k']:>5}"
            f"{c['avg_bits']:>9.4f}{c['payload_bits']:>10}"
            f"{c['mse_agg']:>10.6f}{c['mse_protected']:>10.6f}"
            f"{c['mse_regular']:>10.6f}"
            f"{c['uniform_mse_eq']:>10.6f}{c['ratio']:>8.4f}"
        )
    print()
    for c in res["configs"]:
        (b0, m0), (b1, m1) = c["anchor0"], c["anchor1"]
        print(f"  frac={c['frac']:.2%} ({c['b_lo']},{c['b_hi']}): "
              f"uniform anchors uni({b0})={m0:.6f} uni({b1})={m1:.6f} -> "
              f"eq@rate {c['avg_bits']:.4f} = {c['uniform_mse_eq']:.6f}")
    print()
    print("uni_eq: uniform MSE at the exact same payload budget as the "
          "protected pool,")
    print("  interpolated log-linearly in rate between the two measured "
          "integer anchors")
    print("ratio: pool_mse / uni_eq; <1 means the protected pool reconstructs "
          "the pool better")
    print("  at the same budget. prot_mse/reg_mse are the per-class MSEs "
          "(sink vs bulk).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
