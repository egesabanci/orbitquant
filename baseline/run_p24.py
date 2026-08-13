"""P2.4 — Variable-size residual QJL runner.

Searches the split of the bit budget B = b_mse*d + m between the scalar MSE
quantizer and the residual sketch, and reports the estimator variance vs the
sketch dimension m (fixed x,y with nonzero dot, independent rotation+sketch
per trial).

Usage:
    python3 -m baseline.run_p24 --d 64 --nrot 2000 [--B 128] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import codebooks as cb
from . import p24_varsize as vs
from . import protocol as pr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--B", type=int, default=None,
                    help="total bit budget b_mse*d + m (default 2*d)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d, n_rot = args.d, args.nrot
    B = args.B if args.B is not None else 2 * d
    rng = np.random.default_rng(args.seed)
    x = pr.fixed_vector(d)
    y = pr.fixed_vector(d, kind="all_equal")
    true = float(np.dot(y, x))

    # codebooks built once per bit-width actually used
    b_used = set()
    m_values = [max(1, d // 4), max(1, d // 2), d, 2 * d]
    for m in m_values:
        if m <= B:
            b_used.add(vs.split_budget(B, m, d))
    for b_mse in range(0, min(B // d, vs.B_MAX_BITS) + 1):
        b_used.add(b_mse)
    b_used.discard(0)
    codebooks = {b: cb.beta_lloyd_max(d, b) for b in b_used}

    print(f"P2.4 variable-size residual QJL  d={d} nrot={n_rot} B={B} seed={args.seed}")
    print(f"  true <y,x> = {true:.4f}   (x = e1, y = all_equal)")
    print()

    # --- variance vs sketch dimension m, best split under budget ---
    print(f"{'m':>6}{'b_mse':>7}{'budget':>8}{'bias':>12}{'variance':>12}")
    print("-" * 45)
    rows = []
    for m in m_values:
        if m > B:
            continue
        b_mse = vs.split_budget(B, m, d)
        bias, var = vs.residual_qjl_stats(x, y, b_mse, m, codebooks, n_rot, rng)
        used = b_mse * d + m
        rows.append((var, m, b_mse, used, bias))
        print(f"{m:>6}{b_mse:>7}{used:>8}{bias:>12.4f}{var:>12.5f}")
    best_var, best_m, best_b, best_used, _ = min(rows)
    print(f"  best split in sweep: m={best_m} b_mse={best_b} "
          f"(budget {best_used}) variance={best_var:.5f}")
    print()

    # --- exact-budget Pareto splits (b_mse*d + m == B) ---
    print("exact-budget splits (b_mse*d + m == B):")
    print(f"{'b_mse':>7}{'m':>6}{'bias':>12}{'variance':>12}")
    print("-" * 37)
    exact = []
    for b_mse in range(0, min(B // d, vs.B_MAX_BITS) + 1):
        m = B - b_mse * d
        bias, var = vs.residual_qjl_stats(x, y, b_mse, m, codebooks, n_rot, rng)
        exact.append((var, b_mse, m, bias))
        print(f"{b_mse:>7}{m:>6}{bias:>12.4f}{var:>12.5f}")
    evar, eb, em, _ = min(exact)
    print(f"  best exact split: b_mse={eb} m={em} variance={evar:.5f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
