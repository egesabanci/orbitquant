"""P3.5 -- Block-structured hybrid rotation benchmark.

Compares the block-structured hybrid rotation (fast randomized FWHT ``hdhdh``
followed by a small dense block mixing layer, e.g. 8x8 or 16x16 orthogonal
blocks) against the pure FWHT (``hdhdh``) and the Haar ideal on the rotation
checks: coordinate Beta KS statistic, RMS off-diagonal correlation, and
quantization MSE (plus the inner-product bias ratio and 4th-moment factor gap).

All rotations are ``Rotation`` objects (forward/inverse). Fixed adversarial
vector x = e1, averaged over independent rotations per trial.

Usage:
    python3 -m baseline.run_p35 --d 64 --nrot 2000
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import codebooks as cb
from . import protocol as pr
from . import rotations as rot
from .p35_hybrid import block_mixing_rotation


def build_rotation(name: str, d: int, rng: np.random.Generator, block: int) -> rot.Rotation:
    """Rotation factory: Haar / pure FWHT by name, hybrid by block size."""
    if name == "hybrid":
        return block_mixing_rotation(d, rng, block=block)
    return rot.rotation_from_name(name, d, rng)


def orthogonality_check(R: rot.Rotation, d: int) -> tuple[float, float]:
    """Max norm deviation and inverse-composition error for a Rotation."""
    x = pr.fixed_vector(d)
    y = R.forward(x)
    norm_dev = float(abs(np.linalg.norm(y) - 1.0))
    inv_err = float(np.max(np.abs(R.inverse(y) - x)))
    return norm_dev, inv_err


def eval_rotation(
    label: str,
    d: int,
    n_rot: int,
    rng: np.random.Generator,
    b: int = 2,
    block: int = 8,
) -> dict:
    """Rotation checks for one transform: beta_ks, rms_corr, mse (+ extras).

    MSE base uses a (b-1)-bit exact-Beta Lloyd-Max codebook (TurboQuant
    Algorithm 2: total = (b-1) + 1 = b bits), matching run_rotations.py.
    """
    x = pr.fixed_vector(d)
    y = pr.fixed_vector(d, kind="all_equal")
    cbk = cb.beta_lloyd_max(d, b - 1)

    samples = np.empty(n_rot * d)
    mse = 0.0
    bias_ests = np.empty(n_rot)
    t0 = time.perf_counter()
    for i in range(n_rot):
        R = build_rotation(label, d, rng, block)
        yx = R.forward(x)
        samples[i * d : (i + 1) * d] = yx
        idx = cb.quantize(yx, cbk)
        yhat = cb.dequantize(idx, cbk)
        mse += float(np.sum((yx - yhat) ** 2))
        bias_ests[i] = np.dot(y, R.inverse(yhat))
    elapsed = time.perf_counter() - t0

    ks = pr.beta_ks(d, samples)
    mse /= n_rot
    Y = samples.reshape(n_rot, d)
    corr = np.corrcoef(Y, rowvar=False)
    off = corr - np.eye(d)
    rms_off = float(np.sqrt(np.mean(off**2)))
    c1, c2 = Y[:, 0], Y[:, 1]
    lhs = np.mean(c1**2 * c2**2)
    rhs = np.mean(c1**2) * np.mean(c2**2)
    factor_gap = float(abs(lhs - rhs) / (rhs + 1e-12))
    true = float(np.dot(y, x))
    bias_ratio = float(np.mean(bias_ests) / true)

    return {
        "rotation": R.name,
        "beta_ks": ks,
        "rms_corr": rms_off,
        "factor_gap": factor_gap,
        "mse": mse,
        "bias_ratio": bias_ratio,
        "runtime_s": elapsed,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--b", type=int, default=2,
                    help="total bit budget; MSE base uses b-1 bits (as run_rotations)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", nargs="*", type=int, default=[8, 16],
                    help="dense mixing block sizes to test (must divide d)")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    d, n_rot, b = args.d, args.nrot, args.b

    # quick orthogonality self-check on the hybrid before the benchmark
    max_norm_dev, max_inv_err = 0.0, 0.0
    for block in args.blocks:
        R = block_mixing_rotation(d, rng, block=block)
        nd, ie = orthogonality_check(R, d)
        max_norm_dev = max(max_norm_dev, nd)
        max_inv_err = max(max_inv_err, ie)
    print(f"P3.5 Block-structured hybrid rotation  d={d} nrot={n_rot} b={b} seed={args.seed}")
    print(f"  orthogonality self-check: max ||R x||/||x|| dev={max_norm_dev:.2e}, "
          f"max ||R^-1 R x - x||={max_inv_err:.2e}")

    # rotation table: Haar ideal, pure FWHT, hybrid per block size
    rows = [("haar", 0), ("hdhdh", 0)] + [("hybrid", blk) for blk in args.blocks]
    print(f"{'rotation':<12}{'beta_ks':>9}{'rms_corr':>10}{'fact_gap':>9}"
          f"{'mse':>9}{'bias':>9}{'rt_s':>7}")
    print("-" * 66)
    results = []
    for name, block in rows:
        r = eval_rotation(name, d, n_rot, rng, b=b, block=block)
        results.append(r)
        print(f"{r['rotation']:<12}{r['beta_ks']:>9.4f}{r['rms_corr']:>10.4f}"
              f"{r['factor_gap']:>9.4f}{r['mse']:>9.4f}{r['bias_ratio']:>9.4f}"
              f"{r['runtime_s']:>7.3f}")

    print("-" * 66)
    print(f"Reference (Haar, d=64, b={b}): beta_ks~0.002, rms_corr~0.02, "
          "mse~0.36, bias~0.637")
    return 0


if __name__ == "__main__":
    sys.exit(main())
