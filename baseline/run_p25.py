"""P2.5 -- Covariance-aware rotation benchmark.

Compares covariance-aware rotations -- calibrated offline from an empirical
covariance target (P2.5, proposals.md) -- against the Haar ideal on the
rotation checks (beta_ks, rms_corr, mse).

Two fixed-rotation conventions are benchmarked (see p25_covariance.py for
the full transpose/order explanation):

  covUHP  T = U H D1 P U^T   the proposal's "U*H*P" product structure:
                            decorrelate into the eigenbasis, balancing
                            permutation, Hadamard mix, map back through U.
  covHP   T = H D1 P U^T    transpose/order convention: decorrelate,
                            balance, mix, and stay in the mixed basis --
                            transformed-covariance diagonal is exactly
                            trace(Cov)/d (guaranteed balanced coordinates).

Both are fixed objects: U, P and D1 are derived once and frozen; the same
Rotation is then evaluated on fresh data. Haar follows the shared protocol
(fresh independent rotation per trial, fixed adversarial x = e1).

Tables:
  1. Fixed adversarial x = e1 (shared protocol): Haar full checks; covUHP /
     covHP reported as single fixed-rotation MSE (beta_ks/rms_corr/bias are
     distributional and not defined for one fixed rotation).
  2. Unit vectors drawn from the synthetic covariance (matched data): all
     rotations over data draws -- Haar fresh per trial, covUHP/covHP the
     single calibrated rotations -- pooled for beta_ks/rms_corr/mse.
  3. Transformed-covariance diagonal balance (matched data): per-coordinate
     variance of y = T x, exact (basis-vector T matrix applied to the
     empirical data covariance) for the fixed rotations, expectation over
     the Haar family for Haar.

MSE uses the (b-1)-bit exact-Beta Lloyd-Max codebook (as run_rotations.py).

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
    calibrate_covariance_rotation,
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

    Per trial: vector x from ``sampler()``, a rotation from ``build(rng)``
    (fresh per trial for randomized families, or a closure returning one
    fixed calibrated Rotation for covUHP/covHP), the rotated vector
    quantized with the (b-1)-bit exact-Beta Lloyd-Max codebook (matching
    run_rotations.py / run_p35.py). Coordinate samples are pooled over all
    trials for beta_ks / rms_corr / factor_gap. The bias ratio
    E[<y, R^-1 Q(R x)>]/<y, x> (y = all-equal probe) is only meaningful when
    x is fixed across trials, so it is reported only when ``bias=True``.
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
                    help="largest outlier eigenvalue ratio (halving per channel)")
    ap.add_argument("--ncal", type=int, default=2048,
                    help="calibration sample count for the empirical covariance")
    ap.add_argument("--ncheck", type=int, default=20000,
                    help="draws for the transformed-covariance balance check")
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
          f"{args.kout} outlier channels (largest x{args.ratio:.0f}, halving)")
    print(f"  empirical eigenvalues (top6): {np.round(eigs[:6], 3)}, "
          f"top-{args.kout} variance fraction = {frac:.3f}")

    # --- calibrate the FIXED covariance-aware rotations (once, offline) ---- #
    # covUHP = U H D1 P U^T (proposal's U*H*P product), covHP = H D1 P U^T
    # (transpose/order convention, guaranteed diagonal balance).
    R_uhp = calibrate_covariance_rotation(eigs, U, rng, convention="uhp")
    R_hp = calibrate_covariance_rotation(eigs, U, rng, convention="hp")
    for R in (R_uhp, R_hp):
        nd, ie = orthogonality_check(R, d)
        print(f"  calibrated fixed rotation ({R.name}): max ||R x|| dev={nd:.2e}, "
              f"max ||R^-1 R x - x||={ie:.2e}")

    def build_haar(r):
        return rot.haar(d, r)

    def matched_x():
        z = rng.standard_normal(d)
        x = Q @ (np.sqrt(lam) * z)
        return x / np.linalg.norm(x)

    # --- transformed-covariance diagonal balance (matched data) ------------ #
    # Fixed rotations: exact T matrix (via basis vectors) applied to the
    # empirical data covariance, so the max/min ratio is a property of the
    # deployed rotation itself. Haar: expectation over the randomized family.
    S = np.zeros((d, d))
    for _ in range(args.ncheck):
        x = matched_x()
        S += np.outer(x, x)
    S /= args.ncheck

    def diag_ratio_fixed(R):
        Tm = np.column_stack([R.forward(e) for e in np.eye(d)])
        diag = np.diag(Tm @ S @ Tm.T)
        return float(diag.mean()), float(diag.max() / diag.min())

    vars_h = np.zeros(d)
    for _ in range(args.ncheck):
        vars_h += rot.haar(d, rng).forward(matched_x()) ** 2
    vars_h /= args.ncheck

    print(f"\nTransformed-covariance diagonal balance (matched data, "
          f"{args.ncheck} draws): per-coordinate var of y = T x")
    for R in (R_hp, R_uhp):
        mean_v, ratio = diag_ratio_fixed(R)
        print(f"  {R.name:<8} (single fixed calibrated T): mean var={mean_v:.4f}   "
              f"max/min diag ratio={ratio:.3f}")
    print(f"  haar    (expectation over family):  mean var={vars_h.mean():.4f}   "
          f"max/min diag ratio={vars_h.max() / vars_h.min():.3f}")

    rows = [("haar", build_haar),
            ("covHP", lambda r: R_hp),
            ("covUHP", lambda r: R_uhp)]
    header = (f"{'rotation':<10}{'beta_ks':>9}{'rms_corr':>10}{'fact_gap':>9}"
              f"{'mse':>9}{'bias':>9}{'rt_s':>7}")

    # --- Table 1: fixed adversarial e1 (shared protocol) ------------------- #
    e1 = pr.fixed_vector(d)
    cbk = cb.beta_lloyd_max(d, b - 1)
    print(f"\nRotation checks, fixed adversarial x = e1 (protocol), "
          f"pooled over {n_rot} trials:")
    print(header)
    print("-" * len(header))
    r = eval_rotation("haar", build_haar, d, n_rot, rng, b, sampler=lambda: e1)
    print(f"{r['rotation']:<10}{r['beta_ks']:>9.4f}{r['rms_corr']:>10.4f}"
          f"{r['factor_gap']:>9.4f}{r['mse']:>9.4f}{r['bias_ratio']:>9.4f}"
          f"{r['runtime_s']:>7.3f}")
    # covUHP/covHP are single fixed rotations: distributional checks are not
    # defined; the single-vector MSE of T e1 is deterministic, reported
    # directly.
    for R in (R_hp, R_uhp):
        yx = R.forward(e1)
        yhat = cb.dequantize(cb.quantize(yx, cbk), cbk)
        mse_single = float(np.sum((yx - yhat) ** 2))
        print(f"{R.name:<10}{'n/a':>9}{'n/a':>10}{'n/a':>9}{mse_single:>9.4f}"
              f"{'n/a':>9}{'n/a':>7}")
    print("  (covUHP/covHP: fixed calibrated rotations; beta_ks/rms_corr/"
          "factor_gap/bias are distributional and n/a; MSE is the "
          "single-vector value)")

    # --- Table 2: covariance-matched data ---------------------------------- #
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
