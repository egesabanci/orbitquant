"""P2.5 -- Covariance-aware rotation benchmark.

Compares the covariance-aware rotation T = D2 U H D1 P (empirical eigenbasis
+ Hadamard + balancing permutation) against the Haar ideal on the rotation
checks (beta_ks, rms_corr, mse), following the shared protocol: fixed
adversarial x = e1, averaged over independent rotations per trial, with the
(b-1)-bit exact-Beta Lloyd-Max codebook for MSE.

A second table repeats the same checks on unit vectors drawn from the
synthetic covariance (a few large outlier channels), where the adaptive
rotation's energy balancing is expected to help.

All rotations are ``Rotation`` objects (forward/inverse).

Usage:
    python3 -m baseline.run_p25 --d 64 --nrot 2000
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import codebooks as cb
from . import protocol as pr
from . import rotations as rot
from .p25_covariance import (
    covariance_rotation,
    empirical_eigenbasis,
    synthetic_covariance_data,
    top_k_variance_fraction,
)


def orthogonality_check(R: rot.Rotation, d: int) -> tuple[float, float]:
    """Max norm deviation and inverse-composition error for a Rotation."""
    x = pr.fixed_vector(d)
    y = R.forward(x)
    norm_dev = float(abs(np.linalg.norm(y) - 1.0))
    inv_err = float(np.max(np.abs(R.inverse(y) - x)))
    return norm_dev, inv_err


def eval_rotation(
    label: str,
    build,
    d: int,
    n_rot: int,
    rng: np.random.Generator,
    b: int,
    sampler,
    bias: bool = True,
) -> dict:
    """Rotation checks for one transform: beta_ks, rms_corr, mse (+ extras).

    Per trial: fixed vector x from ``sampler()``, a fresh rotation from
    ``build(rng)``, the rotated vector quantized with the (b-1)-bit
    exact-Beta Lloyd-Max codebook (matching run_rotations.py / run_p35.py).
    Coordinate samples are pooled over all trials for beta_ks / rms_corr /
    factor_gap. The bias ratio E[<y, R^-1 Q(R x)>]/<y, x> (y = all-equal
    probe) is only meaningful when x is fixed across trials, so it is
    reported only when ``bias=True``.
    """
    cbk = cb.beta_lloyd_max(d, b - 1)
    y_probe = pr.fixed_vector(d, kind="all_equal")

    samples = np.empty(n_rot * d)
    mse = 0.0
    bias_ests = np.empty(n_rot) if bias else None
    t0 = time.perf_counter()
    for i in range(n_rot):
        R = build(rng)
        x = sampler()
        yx = R.forward(x)
        samples[i * d : (i + 1) * d] = yx
        idx = cb.quantize(yx, cbk)
        yhat = cb.dequantize(idx, cbk)
        mse += float(np.sum((yx - yhat) ** 2))
        if bias:
            bias_ests[i] = np.dot(y_probe, R.inverse(yhat))
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

    result = {
        "rotation": label,
        "beta_ks": ks,
        "rms_corr": rms_off,
        "factor_gap": factor_gap,
        "mse": mse,
        "runtime_s": elapsed,
    }
    if bias:
        true = float(np.dot(y_probe, sampler()))
        result["bias_ratio"] = float(np.mean(bias_ests) / true)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--b", type=int, default=2,
                    help="total bit budget; MSE base uses b-1 bits (as run_rotations)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kout", type=int, default=4,
                    help="outlier channels in the synthetic covariance")
    ap.add_argument("--ratio", type=float, default=100.0,
                    help="outlier eigenvalue ratio")
    ap.add_argument("--ncal", type=int, default=2048,
                    help="calibration sample count for the empirical covariance")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    d, n_rot, b = args.d, args.nrot, args.b

    # --- synthetic covariance (empirical, from calibration data) ----------- #
    X, Sigma, Q, lam = synthetic_covariance_data(
        d, rng, n_cal=args.ncal, k_out=args.kout, out_ratio=args.ratio
    )
    eigs, U = empirical_eigenbasis(Sigma)
    frac = top_k_variance_fraction(eigs, args.kout)
    print(f"P2.5 Covariance-aware rotation  d={d} nrot={n_rot} b={b} seed={args.seed}")
    print(f"  synthetic covariance: {args.ncal} calibration samples, "
          f"{args.kout} outlier channels x{args.ratio:.0f}")
    print(f"  empirical eigenvalues (top6): {np.round(eigs[:6], 3)}, "
          f"top-{args.kout} variance fraction = {frac:.3f}")

    # --- orthogonality self-check on the derived rotation ------------------ #
    max_nd, max_ie = 0.0, 0.0
    for _ in range(3):
        nd, ie = orthogonality_check(covariance_rotation(eigs, U, rng), d)
        max_nd = max(max_nd, nd)
        max_ie = max(max_ie, ie)
    print(f"  orthogonality self-check (covUHP): max ||R x|| dev={max_nd:.2e}, "
          f"max ||R^-1 R x - x||={max_ie:.2e}")

    def build_haar(r):
        return rot.haar(d, r)

    def build_cov(r):
        return covariance_rotation(eigs, U, r)

    rows = [("haar", build_haar), ("covUHP", build_cov)]
    header = (f"{'rotation':<10}{'beta_ks':>9}{'rms_corr':>10}{'fact_gap':>9}"
              f"{'mse':>9}{'bias':>9}{'rt_s':>7}")

    # --- Table 1: fixed adversarial e1 (shared protocol) ------------------- #
    e1 = pr.fixed_vector(d)
    print(f"\nRotation checks, fixed adversarial x = e1 (protocol), "
          f"pooled over {n_rot} trials:")
    print(header)
    print("-" * len(header))
    for name, build in rows:
        r = eval_rotation(name, build, d, n_rot, rng, b, sampler=lambda: e1)
        print(f"{r['rotation']:<10}{r['beta_ks']:>9.4f}{r['rms_corr']:>10.4f}"
              f"{r['factor_gap']:>9.4f}{r['mse']:>9.4f}{r['bias_ratio']:>9.4f}"
              f"{r['runtime_s']:>7.3f}")

    # --- Table 2: covariance-matched data ---------------------------------- #
    def matched_x():
        z = rng.standard_normal(d)
        x = Q @ (np.sqrt(lam) * z)
        return x / np.linalg.norm(x)

    print(f"\nRotation checks, unit vectors drawn from the synthetic "
          f"covariance (matched data), pooled over {n_rot} trials:")
    print(header)
    print("-" * len(header))
    for name, build in rows:
        r = eval_rotation(name, build, d, n_rot, rng, b,
                          sampler=matched_x, bias=False)
        print(f"{r['rotation']:<10}{r['beta_ks']:>9.4f}{r['rms_corr']:>10.4f}"
              f"{r['factor_gap']:>9.4f}{r['mse']:>9.4f}{'n/a':>9}"
              f"{r['runtime_s']:>7.3f}")

    print("-" * len(header))
    print(f"Reference (Haar, d=64, b={b}): beta_ks~0.002, rms_corr~0.02, "
          "mse~0.36, bias~0.637")
    return 0


if __name__ == "__main__":
    sys.exit(main())
