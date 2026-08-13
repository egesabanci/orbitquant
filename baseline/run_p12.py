"""P1.2 — Asymmetric key/value bit budgets (runner).

Benchmarks asymmetric (K, V) bit splits at the same average budget:

- Keys: (K-1)-bit MSE base + 1-bit QJL residual product estimator, K
  bits/coordinate -> key inner-product error (bias, bias ratio, RMSE).
- Values: V-bit Lloyd-Max MSE codebook, V bits/coordinate -> value MSE.

Usage:
    python3 -m baseline.run_p12 --d 64 --nrot 2000
    python3 -m baseline.run_p12 --d 64 --nrot 2000 --splits "4,2;2,4;3,3"
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import p12_key_value as p12


def parse_splits(s: str) -> list[tuple[int, int]]:
    """Parse "K,V;K,V" into [(K, V), ...]."""
    splits = []
    for part in s.split(";"):
        if not part.strip():
            continue
        k, v = part.split(",")
        splits.append((int(k), int(v)))
    if not splits:
        raise ValueError(f"no splits parsed from {s!r}")
    return splits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument(
        "--splits",
        type=str,
        default="4,2;2,4;3,3",
        help="semicolon-separated K,V splits (default 4,2;2,4;3,3)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d, n_rot = args.d, args.nrot
    splits = parse_splits(args.splits)
    rng = np.random.default_rng(args.seed)

    print(f"P1.2 key/value asymmetric budgets  d={d} nrot={n_rot} seed={args.seed}")
    print("keys: (K-1)-bit MSE base + 1-bit QJL residual (K bits/coord)")
    print("values: V-bit Lloyd-Max MSE codebook (V bits/coord)")
    avgs = {(k + v) / 2 for k, v in splits}
    if len(avgs) > 1:
        print(f"NOTE: splits have different average budgets {sorted(avgs)} -- "
              "not directly comparable")
    print()

    header = (
        f"{'split':<9}{'avg_bits':>9}{'total_bits':>11}"
        f"{'key_bias':>10}{'key_ratio':>10}{'key_rmse':>10}"
        f"{'val_mse':>10}{'val_floor':>10}"
    )
    print(header)
    print("-" * len(header))
    for k_bits, v_bits in splits:
        t0 = time.perf_counter()
        res = p12.benchmark_split(d, k_bits, v_bits, n_rot, rng)
        elapsed = time.perf_counter() - t0
        print(
            f"K={res['K']},V={res['V']:<5}"
            f"{(res['K'] + res['V']) / 2:>9.1f}"
            f"{res['total_bits']:>11}"
            f"{res['key_bias']:>10.5f}"
            f"{res['key_bias_ratio']:>10.4f}"
            f"{res['key_rmse']:>10.5f}"
            f"{res['value_mse']:>10.5f}"
            f"{res['value_floor']:>10.5f}"
            f"   ({elapsed:.1f}s)"
        )
    print()
    print("key_bias: E[<y,Q(x)>] - <y,x>; key_ratio: E[<y,Q(x)>]/<y,x> (unbiased ~1.0)")
    print("key_rmse: sqrt(E[(<y,Q(x)> - <y,x>)^2]); val_mse: E[||x - Q(x)||^2]")
    print("val_floor: 1/4^V Lloyd-Max distortion floor (per-coordinate basis)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
