"""P2.8 -- OCTOPUS-style triplet codec benchmark.

Compares the triplet codec (norm + two octahedral direction coordinates per
contiguous triplet, Lloyd-Max quantized with a non-uniform bit allocation)
against the scalar exact-Beta Lloyd-Max quantizer at matched total bits:
every split with b_norm + 2*b_dir == 3*b uses exactly b*d bits, the same as
the b-bit scalar reference. Fixed adversarial x = e1, independent Haar
rotations per trial; MSE is the per-vector squared reconstruction error.

Usage:
    python3 -m baseline.run_p28 --d 64 --nrot 2000 [--b 1 2 3] [--ncal 1000] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import p28_octopus as p28
from . import protocol as pr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--b", nargs="*", type=int, default=[1, 2, 3],
                    help="per-coordinate bit budgets to compare")
    ap.add_argument("--ncal", type=int, default=1000,
                    help="rotations used to calibrate the codebooks")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d, n_rot, n_cal = args.d, args.nrot, args.ncal
    b_values = args.b
    if d < 3:
        ap.error("--d must be >= 3")
    for b in b_values:
        if b < 1:
            ap.error("--b values must be >= 1")

    rng = np.random.default_rng(args.seed)
    n_t, rem = d // 3, d % 3

    # codebook calibration is shared across splits (distributions depend on d)
    t0 = time.perf_counter()
    norms, coords = p28.calibration_samples(d, n_cal, args.seed)
    cal_time = time.perf_counter() - t0

    print(f"P2.8 OCTOPUS-style triplet codec  d={d} nrot={n_rot} ncal={n_cal} "
          f"seed={args.seed}")
    print(f"  x = {pr.fixed_vector(d, 'e1')[:1].tolist()} (e1), independent "
          f"sign-corrected Haar rotations per trial")
    print(f"  triplets={n_t} leftover coords={rem} (scalar b bits each); "
          f"calibration {cal_time:.1f}s")
    print(f"  total bits matched: b*d for both schemes (b_norm + 2*b_dir == 3*b "
          "per triplet)")
    print()

    worst_ratio = 0.0
    for b in b_values:
        cbk_ref = None  # scalar reference codebook (built lazily via scalar_mse)
        ms = p28.scalar_mse(d, n_rot, b, rng)
        splits = p28.matched_splits(b)
        print(f"b={b}  (budget {b * d} bits; scalar Lloyd-Max MSE = {ms:.4f})")
        print(f"  {'b_norm':>6}{'b_dir':>6}{'bits/trip':>10}{'MSE':>10}"
              f"{'ratio':>8}")
        print("  " + "-" * 38)
        best = None
        for b_norm, b_dir in splits:
            cbs = p28.codebooks_from_samples(d, b, b_norm, b_dir, norms, coords)
            mt = p28.triplet_codec_mse(d, n_rot, cbs, rng)
            ratio = mt / ms
            worst_ratio = max(worst_ratio, ratio)
            if best is None or mt < best[0]:
                best = (mt, b_norm, b_dir, ratio)
            print(f"  {b_norm:>6}{b_dir:>6}{b_norm + 2 * b_dir:>10}"
                  f"{mt:>10.4f}{ratio:>8.3f}")
        m_best, bn_best, bd_best, ratio_best = best
        print(f"  best split: b_norm={bn_best} b_dir={bd_best} -> MSE {m_best:.4f} "
              f"({ratio_best:.2f}x scalar)")
        print()

    print(f"note: measured values only -- no pass/fail thresholds. Scalar "
          f"exact-Beta Lloyd-Max is the distortion-optimal reference "
          f"(worst triplet/scalar ratio seen: {worst_ratio:.2f}x).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
