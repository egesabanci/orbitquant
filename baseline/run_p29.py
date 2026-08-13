"""P2.9 — Joint rounding for small-block codecs runner.

Compares per-coordinate MSE of independent (per-coordinate nearest centroid)
vs joint (tiny local candidate enumeration around the nearest centroids)
rounding for the spherical block codec of P2.9, at block sizes 2 (2-coordinate
block) and 3 (the OCTOPUS-shaped triplet: norm + 2 direction coords,
renormalized onto the sphere) and direction bit-widths 1..3.

Usage:
    python3 -m baseline.run_p29 --d 64 --nrot 2000
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import codebooks as cb
from . import p29_joint_round as jr
from . import protocol as pr


def scalar_reference_mse(d: int, n_rot: int, rng: np.random.Generator, bits) -> np.ndarray:
    """Per-coordinate MSE of plain independent scalar Beta Lloyd-Max (context).

    Same protocol: fixed x = e1, independent Haar rotations per trial.
    """
    x = pr.fixed_vector(d)
    codebooks = {b: cb.beta_lloyd_max(d, b) for b in bits}
    mse = np.zeros(len(bits))
    for _ in range(n_rot):
        y = pr.random_rotation(d, rng) @ x
        for j, b in enumerate(bits):
            cbk = codebooks[b]
            mse[j] += float(np.sum((y - cb.dequantize(cb.quantize(y, cbk), cbk)) ** 2))
    return mse / (n_rot * d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--b_dir", nargs="*", type=int, default=[1, 2, 3],
                    help="direction codebook bit-widths to test")
    ap.add_argument("--b_norm", type=int, default=4, help="norm codebook bit-width")
    ap.add_argument("--k", nargs="*", type=int, default=[1, 2],
                    help="local candidate radii (+/- k per coordinate)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    d, n_rot = args.d, args.nrot
    b_dirs = sorted(set(args.b_dir))
    ks = sorted(set(args.k))
    t_start = time.perf_counter()

    print(f"P2.9 joint rounding for small-block codecs  d={d} nrot={n_rot} seed={args.seed}")
    print("codec: spherical block codec  block -> (norm, direction); direction")
    print("coords scalar-quantized (finite-d Beta Lloyd-Max), reconstruction")
    print("renormalizes the direction onto the sphere (couples the scalars).")
    print()

    t0 = time.perf_counter()
    ref = scalar_reference_mse(d, n_rot, rng, b_dirs)
    print("scalar reference (independent Beta Lloyd-Max, no block structure):")
    for b, m in zip(b_dirs, ref):
        print(f"  b={b} ({2**b} levels, {b} bits/coord): mse={m:.5f}")
    print(f"  [elapsed {time.perf_counter() - t0:.1f}s]")
    print()

    print(f"{'block':>5}{'b_dir':>6}{'bits/coord':>10}{'k':>3}"
          f"{'mse_ind':>10}{'mse_joint':>10}{'gain %':>9}")
    print("-" * 53)
    best = None
    for block in (2, 3):
        for b_dir in b_dirs:
            codebooks = jr.spherical_block_codebooks(d, block, b_dir, args.b_norm, rng)
            t0 = time.perf_counter()
            mse_i, mse_j = jr.block_codec_mse(d, block, codebooks, n_rot, rng, ks)
            bits_per = (args.b_norm + block * b_dir) / block
            for k in ks:
                mj = mse_j[k]
                gain = 100.0 * (mse_i - mj) / mse_i
                if best is None or gain > best[0]:
                    best = (gain, block, b_dir, k)
                print(f"{block:>5}{b_dir:>6}{bits_per:>10.3f}{k:>3}"
                      f"{mse_i:>10.5f}{mj:>10.5f}{gain:>9.3f}")
            print(f"  [block={block} b_dir={b_dir} elapsed {time.perf_counter() - t0:.1f}s]")
    print("-" * 53)
    if best is not None:
        g, blk, bd, k = best
        print(f"best joint-rounding gain: {g:.3f}% at block={blk} b_dir={bd} k={k}")
    print(f"total elapsed {time.perf_counter() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
