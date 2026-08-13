"""P1.4 -- Outlier-aware channel permutation benchmark.

Compares three channel orderings applied before a Hadamard+sign-flip rotation
(``rot.hadamard_sign_flip``, rounds=1 -> "hd") on the rotation checks
(beta_ks, rms_corr, mse):

- no-perm:      identity channel order (outlier channels stay clustered);
- random-perm:  a fresh random permutation per trial;
- outlier-perm: the P1.4 permutation from calibration on synthetic vectors
                with a few outlier channels, fixed after calibration,
                spreading the top outlier channels one per FWHT butterfly
                group.

All rotations are ``Rotation`` objects (forward/inverse). Fixed adversarial
x, averaged over independent rotations per trial. The primary fixed x is a
deterministic unit vector with clustered outlier channels (the P1.4-relevant
case); x = e1 is also reported -- H P e1 has flat |coordinates| for every
permutation P, so e1 cannot separate the permutation variants.

Usage:
    python3 -m baseline.run_p14 --d 64 --nrot 2000
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import codebooks as cb
from . import protocol as pr
from . import rotations as rot
from .p14_outlier_perm import (
    calibration_vectors,
    channel_scores,
    outlier_aware_permutation,
    outlier_vector,
    permuted_hadamard_rotation,
)


def orthogonality_check(R: rot.Rotation, d: int) -> tuple[float, float]:
    """Max norm deviation and inverse-composition error for a Rotation."""
    x = pr.fixed_vector(d)
    y = R.forward(x)
    norm_dev = float(abs(np.linalg.norm(y) - 1.0))
    inv_err = float(np.max(np.abs(R.inverse(y) - x)))
    return norm_dev, inv_err


def eval_variant(
    label: str,
    d: int,
    n_rot: int,
    rng: np.random.Generator,
    x: np.ndarray,
    cbk: np.ndarray,
    make_rotation,
) -> dict:
    """Rotation checks for one variant: beta_ks, rms_corr, mse.

    MSE base uses a (b-1)-bit exact-Beta Lloyd-Max codebook (TurboQuant
    Algorithm 2: total = (b-1) + 1 = b bits), as in run_rotations.py/run_p35.
    MSE is computed in the rotated domain (identical for orthogonal R).
    """
    samples = np.empty(n_rot * d)
    mse = 0.0
    t0 = time.perf_counter()
    for i in range(n_rot):
        R = make_rotation()
        yx = R.forward(x)
        samples[i * d : (i + 1) * d] = yx
        idx = cb.quantize(yx, cbk)
        yhat = cb.dequantize(idx, cbk)
        mse += float(np.sum((yx - yhat) ** 2))
    elapsed = time.perf_counter() - t0

    ks = pr.beta_ks(d, samples)
    Y = samples.reshape(n_rot, d)
    corr = np.corrcoef(Y, rowvar=False)
    off = corr - np.eye(d)
    rms_off = float(np.sqrt(np.mean(off**2)))
    return {
        "rotation": label,
        "beta_ks": ks,
        "rms_corr": rms_off,
        "mse": mse / n_rot,
        "runtime_s": elapsed,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--b", type=int, default=2,
                    help="total bit budget; MSE base uses b-1 bits (as run_rotations)")
    ap.add_argument("--n-out", type=int, default=8,
                    help="number of outlier channels")
    ap.add_argument("--gain", type=float, default=4.0,
                    help="outlier channel amplification (x background std)")
    ap.add_argument("--n-cal", type=int, default=512,
                    help="number of synthetic calibration vectors")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d, n_rot, b = args.d, args.nrot, args.b
    rng = np.random.default_rng(args.seed)

    # P1.4 calibration on synthetic vectors with a few outlier channels
    channels = np.arange(args.n_out)  # fixed, clustered outlier channels
    cal = calibration_vectors(d, args.n_cal, channels,
                              out_gain=args.gain, seed=args.seed)
    scores = channel_scores(cal)
    aware_perm = outlier_aware_permutation(d, scores, args.n_out)
    detected = np.argsort(-scores, kind="stable")[: args.n_out]

    # orthogonality self-check on the composed rotations
    max_norm_dev, max_inv_err = 0.0, 0.0
    for R in (
        permuted_hadamard_rotation(d, rng),
        permuted_hadamard_rotation(d, rng, perm=aware_perm),
    ):
        nd, ie = orthogonality_check(R, d)
        max_norm_dev = max(max_norm_dev, nd)
        max_inv_err = max(max_inv_err, ie)

    print(f"P1.4 outlier-aware permutation  d={d} nrot={n_rot} b={b} seed={args.seed}")
    print(f"  calibration: n_cal={args.n_cal} outlier channels={channels.tolist()} "
          f"gain={args.gain:g}")
    print(f"  detected outliers (by score): {detected.tolist()}")
    print(f"  aware perm (channel->position): {aware_perm.tolist()}")
    print(f"  orthogonality self-check: max ||R x||/||x|| dev={max_norm_dev:.2e}, "
          f"max ||R^-1 R x - x||={max_inv_err:.2e}")

    x_out = outlier_vector(d, channels, out_gain=args.gain, seed=args.seed)
    x_e1 = pr.fixed_vector(d)
    cbk = cb.beta_lloyd_max(d, b - 1)

    variants = [
        ("no-perm", lambda: permuted_hadamard_rotation(d, rng)),
        ("random-perm",
         lambda: permuted_hadamard_rotation(d, rng, perm=rng.permutation(d))),
        ("outlier-perm",
         lambda: permuted_hadamard_rotation(d, rng, perm=aware_perm)),
    ]

    for xlabel, x in (
        ("x=outlier (fixed adversarial, clustered outliers)", x_out),
        ("x=e1 (degenerate for permutations)", x_e1),
    ):
        print(f"\nfixed {xlabel}")
        print(f"{'rotation':<14}{'beta_ks':>9}{'rms_corr':>10}{'mse':>9}{'rt_s':>7}")
        print("-" * 52)
        for label, factory in variants:
            r = eval_variant(label, d, n_rot, rng, x, cbk, factory)
            print(f"{r['rotation']:<14}{r['beta_ks']:>9.4f}{r['rms_corr']:>10.4f}"
                  f"{r['mse']:>9.4f}{r['runtime_s']:>7.3f}")

    print("-" * 52)
    print(f"Reference (Haar, d=64, b={b}): beta_ks~0.002, rms_corr~0.02, "
          "mse~0.36 (any fixed unit x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
