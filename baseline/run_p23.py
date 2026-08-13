"""P2.3 — Orthogonalized residual sketches runner.

Compares the QJL inner-product estimator with iid Gaussian sketch rows vs
QR-orthogonalized (exactly orthonormal) rows: bias and variance of <y, x>
over independent sketches per trial, fixed x,y with nonzero dot.

Usage:
    python3 -m baseline.run_p23 --d 64 --nrot 2000 [--m 64] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import p23_orthogonal as og
from . import protocol as pr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--m", type=int, default=None,
                    help="sketch dimension (default d; orthogonalized rows "
                         "require m <= d)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d, n_rot = args.d, args.nrot
    m = args.m if args.m is not None else d
    if m > d:
        raise SystemExit(f"m={m} > d={d}: orthogonalized rows require m <= d")
    rng = np.random.default_rng(args.seed)
    x = pr.fixed_vector(d)
    y = pr.fixed_vector(d, kind="all_equal")
    true = float(np.dot(y, x))

    print(f"P2.3 orthogonalized residual sketches  d={d} nrot={n_rot} "
          f"m={m} seed={args.seed}")
    print(f"  true <y,x> = {true:.4f}   (x = e1, y = all_equal)")
    print()

    # --- main comparison at sketch dimension m ---
    print(f"{'sketch rows':>14}{'bias':>12}{'variance':>12}")
    print("-" * 38)
    rows = {}
    for mode in ("iid", "orth"):
        bias, var = og.qjl_stats(x, y, m, mode, n_rot, rng)
        rows[mode] = (bias, var)
        print(f"{mode:>14}{bias:>12.4f}{var:>12.5f}")
    b_iid, v_iid = rows["iid"]
    b_orth, v_orth = rows["orth"]
    print(f"  variance ratio iid/orth = {v_iid / v_orth:.3f}")
    print()

    # --- variance vs sketch dimension (orthogonalized needs m <= d) ---
    print("variance vs sketch dimension m:")
    print(f"{'m':>6}{'iid var':>12}{'orth var':>12}{'ratio':>8}")
    print("-" * 38)
    for mm in sorted({max(1, d // 4), max(1, d // 2), d, m}):
        if mm > d:
            continue
        _, vi = og.qjl_stats(x, y, mm, "iid", n_rot, rng)
        _, vo = og.qjl_stats(x, y, mm, "orth", n_rot, rng)
        print(f"{mm:>6}{vi:>12.5f}{vo:>12.5f}{vi / vo:>8.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
