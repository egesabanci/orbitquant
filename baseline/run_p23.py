"""P2.3 — Orthogonalized residual sketches runner.

Compares the residual-QJL inner-product estimator (scalar stage + sketch of
the residual, the P2.4 path) with iid Gaussian sketch rows vs
QR-orthogonalized (exactly orthonormal) rows: bias and variance of <y, x>
over independent rotation + sketch per trial, fixed x,y with nonzero dot.

Usage:
    python3 -m baseline.run_p23 --d 64 --nrot 2000 [--m 64] [--b 1] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import codebooks as cb
from . import p23_orthogonal as og
from . import protocol as pr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--m", type=int, default=None,
                    help="sketch dimension (default d; orthogonalized rows "
                         "require m <= d)")
    ap.add_argument("--b", type=int, default=1,
                    help="scalar-stage bit-width (0 = pure QJL, no scalar stage)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d, n_rot = args.d, args.nrot
    m = args.m if args.m is not None else d
    b = args.b
    if m > d:
        raise SystemExit(f"m={m} > d={d}: orthogonalized rows require m <= d")
    rng = np.random.default_rng(args.seed)
    x = pr.fixed_vector(d)
    y = pr.fixed_vector(d, kind="all_equal")
    true = float(np.dot(y, x))

    # codebooks for every scalar bit-width actually measured
    b_used = {b, 0, 1, 2}
    codebooks = {bb: cb.beta_lloyd_max(d, bb) for bb in b_used if bb > 0}

    print(f"P2.3 orthogonalized residual sketches  d={d} nrot={n_rot} "
          f"m={m} b_mse={b} seed={args.seed}")
    print(f"  true <y,x> = {true:.4f}   (x = e1, y = all_equal)")
    print()

    # --- main comparison: residual-QJL stage, iid vs orth rows ---
    print(f"{'sketch rows':>14}{'bias':>12}{'variance':>12}")
    print("-" * 38)
    rows = {}
    for mode in ("iid", "orth"):
        bias, var = og.residual_qjl_stats(x, y, b, m, mode, codebooks, n_rot, rng)
        rows[mode] = (bias, var)
        print(f"{mode:>14}{bias:>12.4f}{var:>12.5f}")
    b_iid, v_iid = rows["iid"]
    b_orth, v_orth = rows["orth"]
    print(f"  variance ratio iid/orth = {v_iid / v_orth:.3f}")
    print()

    # --- variance vs sketch dimension m (orthogonalized needs m <= d) ---
    print("variance vs sketch dimension m:")
    print(f"{'m':>6}{'iid var':>12}{'orth var':>12}{'ratio':>8}")
    print("-" * 38)
    for mm in sorted({max(1, d // 4), max(1, d // 2), d, m}):
        if mm > d:
            continue
        _, vi = og.residual_qjl_stats(x, y, b, mm, "iid", codebooks, n_rot, rng)
        _, vo = og.residual_qjl_stats(x, y, b, mm, "orth", codebooks, n_rot, rng)
        print(f"{mm:>6}{vi:>12.5f}{vo:>12.5f}{vi / vo:>8.3f}")
    print()

    # --- variance vs scalar bit-width b (b = 0: pure QJL ablation) ---
    print("variance vs scalar bit-width b (m fixed):")
    print(f"{'b':>4}{'iid var':>12}{'orth var':>12}{'ratio':>8}")
    print("-" * 38)
    for bb in (0, 1, 2):
        _, vi = og.residual_qjl_stats(x, y, bb, m, "iid", codebooks, n_rot, rng)
        _, vo = og.residual_qjl_stats(x, y, bb, m, "orth", codebooks, n_rot, rng)
        print(f"{bb:>4}{vi:>12.5f}{vo:>12.5f}{vi / vo:>8.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
