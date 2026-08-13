"""P2.7 — Clipped/companded codebook benchmark.

Compares three scalar codebook families at b = 1, 2 bits:

- ``plain``: the exact-Beta Lloyd-Max codebook (``codebooks.beta_lloyd_max``),
  the standard TurboQuant reference.
- ``clipped``: Lloyd-Max on the source density truncated at a calibrated clip
  (heavy tails are collapsed onto the outermost centroid).
- ``companded``: calibrated mu-law companding (log-law is the same family,
  alpha = scale/mu), quantized as the true companding quantizer (uniform
  bins of g(x), inverse-companded bin midpoints).

Protocol note: under the rotation protocol (fixed unit x, independent Haar
rotations) the coordinate law is exactly Beta((d-1)/2, (d-1)/2) — light-tailed
for d = 64 but heavy-tailed (arcsine, U-shaped, infinite density at +/-1) for
d = 2. Tables 2a (d=2) and 2b (d=64) therefore run the full protocol at both
ends of the tail spectrum. Table 1 additionally benchmarks the codebook
families directly on heavy-tailed coordinate sources (an outlier-channel
mixture and Student-t df=3) — the regime where the plain Beta codebook's
assumption fails — explicitly outside the rotation protocol, since a fixed
unit vector under Haar rotations cannot produce heavier-than-Beta coordinates.

Usage:
    python3 -m baseline.run_p27 --d 64 --nrot 2000
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import codebooks as cb
from . import p27_clip as p27
from . import protocol as pr


def mse_on_vectors(samples: np.ndarray, codebook: np.ndarray) -> float:
    """Mean per-vector MSE over (n, d) rotated-coordinate samples."""
    yh = cb.dequantize(cb.quantize(samples, codebook), codebook)
    return float(np.mean(np.sum((samples - yh) ** 2, axis=1)))


def protocol_table(
    label: str,
    d: int,
    n_rot: int,
    rng: np.random.Generator,
    cal_rng: np.random.Generator,
    n_cal: int,
) -> None:
    """Full rotation protocol pass: fixed x=e1, independent Haar rotations.

    The coordinate law is Beta((d-1)/2, (d-1)/2) on [-1,1]: light-tailed for
    large d, heavy-tailed (arcsine, infinite density at +/-1) for d=2. At
    d=2 the plain reference and calibration use the endpoint-safe analytic
    arcsine machinery (the grid Lloyd-Max degenerates on the 1/sqrt(1-x^2)
    spike); otherwise the exact-Beta codebook and sampler are used. Reports
    per-vector MSE (summed over coordinates, averaged over rotations).
    """
    if d == 2:
        den = p27.arcsine_pdf
        cal = p27.arcsine_samples(n_cal, cal_rng)
        plain_of = p27.arcsine_lloyd_max
        clip_of = lambda b, tau: p27.arcsine_clipped_lloyd_max(b, tau)
    else:
        den = lambda z: cb.beta_pdf(z, d)
        cal = pr.sample_beta((d - 1) / 2, (d - 1) / 2, n_cal, cal_rng) * 2 - 1
        plain_of = lambda b: cb.beta_lloyd_max(d, b)
        clip_of = lambda b, tau: p27.clipped_lloyd_max(den, b, tau)
    x = pr.fixed_vector(d)
    rot = np.empty((n_rot, d))
    for i in range(n_rot):
        P = pr.random_rotation(d, rng)
        rot[i] = P @ x

    print(f"{label}: per-vector MSE (x=e1, {n_rot} independent Haar rotations)")
    print(f"{'b':>3}{'plain':>12}{'clipped':>12}{'companded':>12}")
    print("-" * 39)
    for b in (1, 2):
        tau, _, _ = p27.calibrate_clip(
            den, b, cal, builder=lambda _d, bb, tt: clip_of(bb, tt))
        mu, s, _, _ = p27.calibrate_compander(b, cal)
        m_plain = mse_on_vectors(rot, plain_of(b))
        m_clip = mse_on_vectors(rot, clip_of(b, tau))
        m_comp = d * p27.companded_mse(rot.reshape(-1), b, mu, s)
        print(f"{b:>3}{m_plain:>12.4f}{m_clip:>12.4f}{m_comp:>12.4f}")
    tau, _, _ = p27.calibrate_clip(
        den, 1, cal, builder=lambda _d, bb, tt: clip_of(bb, tt))
    mu, s, _, _ = p27.calibrate_compander(1, cal)
    print(f"  calibration (b=1): clip tau*={tau:.3f}  "
          f"compander mu*={mu} s*={s:.3f}")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--ncal", type=int, default=50_000,
                    help="iid samples used to calibrate clip/compander")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d, n_rot = args.d, args.nrot
    n_eval = n_rot * d
    rng = np.random.default_rng(args.seed)
    cal_rng = np.random.default_rng(args.seed + 1000)
    t0 = time.perf_counter()

    # ---- calibration (independent rng) and evaluation samples ------------- #
    cal_mix = p27.heavy_tail_samples(args.ncal, cal_rng)
    cal_t = p27.student_t_samples(args.ncal, cal_rng)

    x_fixed = p27.heavy_tail_samples(d, rng)  # fixed adversarial heavy x
    ev_mix = p27.heavy_tail_samples(n_eval, rng)
    ev_t = p27.student_t_samples(n_eval, rng)

    plain = {b: cb.beta_lloyd_max(d, b) for b in (1, 2)}
    matched_mix = {b: p27.unclipped_lloyd_max(p27.heavy_tail_pdf, b, 40.0)
                   for b in (1, 2)}
    matched_t = {b: p27.unclipped_lloyd_max(p27.student_t_pdf, b, 30.0)
                 for b in (1, 2)}

    print(f"P2.7 clipped/companded codebooks  d={d} nrot={n_rot} "
          f"ncal={args.ncal} seed={args.seed}")
    print("  heavy-tail sources: mixture 98% N(0,1)+2% N(0,100); "
          "Student-t df=3 (unit variance)")
    print(f"  plain = exact-Beta(d={d}) codebook (uncalibrated reference); "
          "matched-LM = Lloyd-Max on the true law")
    print()

    # ---- Table 1: heavy-tailed coordinate sources ------------------------- #
    # Outside the rotation protocol: a fixed unit vector under Haar rotations
    # has exactly Beta coordinates, so heavier-than-Beta tails cannot arise
    # there; this table is the regime where the plain codebook's assumption
    # fails and clipping/companding are designed to help.
    print("Table 1 -- heavy-tailed coordinate sources (coordinate-level "
          "benchmark, outside the rotation protocol)")
    for name, cal, ev, mpdf, mref in [
        ("mixture (outlier channels)", cal_mix, ev_mix,
         p27.heavy_tail_pdf, matched_mix),
        ("student-t df=3", cal_t, ev_t, p27.student_t_pdf, matched_t),
    ]:
        print(f"  {name}: per-coordinate MSE over {n_eval} samples")
        print(f"{'b':>3}{'plain':>12}{'clipped':>12}{'companded':>12}"
              f"{'matched-LM':>12}")
        print("  " + "-" * 49)
        vor = {}
        for b in (1, 2):
            tau, _, _ = p27.calibrate_clip(mpdf, b, cal)
            mu, s, cbk, _ = p27.calibrate_compander(b, cal)
            vor[b] = p27.mse_samples(cbk, ev)
            m_plain = p27.mse_samples(plain[b], ev)
            m_clip = p27.mse_samples(p27.clipped_lloyd_max(mpdf, b, tau), ev)
            m_comp = p27.companded_mse(ev, b, mu, s)
            m_match = p27.mse_samples(mref[b], ev)
            print(f"  {b:>3}{m_plain:>12.4f}{m_clip:>12.4f}{m_comp:>12.4f}"
                  f"{m_match:>12.4f}")
        print(f"    companded nearest-centroid (transformed-level) MSE: "
              f"b=1 {vor[1]:.4f}  b=2 {vor[2]:.4f}")
        tau, _, _ = p27.calibrate_clip(mpdf, 1, cal)
        mu, s, _, _ = p27.calibrate_compander(1, cal)
        print(f"    calibration (b=1): clip tau*={tau:.3f}  compander "
              f"mu*={mu} s*={s:.3f} (log-law alpha*=s/mu={s/mu:.4f})")

    # fixed adversarial heavy-tailed vector (coordinate-level, as above)
    print(f"  fixed adversarial x ({d} coords, mixture draw, "
          f"max|x|={np.max(np.abs(x_fixed)):.2f}): per-coordinate MSE")
    print(f"  {'b':>3}{'plain':>12}{'clipped':>12}{'companded':>12}")
    print("  " + "-" * 37)
    for b in (1, 2):
        tau, _, _ = p27.calibrate_clip(p27.heavy_tail_pdf, b, cal_mix)
        mu, s, _, _ = p27.calibrate_compander(b, cal_mix)
        m_plain = p27.mse_samples(plain[b], x_fixed)
        m_clip = p27.mse_samples(p27.clipped_lloyd_max(
            p27.heavy_tail_pdf, b, tau), x_fixed)
        m_comp = p27.companded_mse(x_fixed, b, mu, s)
        print(f"  {b:>3}{m_plain:>12.4f}{m_clip:>12.4f}{m_comp:>12.4f}")
    print()

    # ---- Tables 2a/2b: full rotation protocol ----------------------------- #
    # d=2: coordinate law is arcsine (Beta(1/2,1/2) on [-1,1]) -- the
    # heavy-tailed end of the protocol's own coordinate law.
    # d=args.d: light-tailed (Beta bell) for d=64.
    protocol_table(
        "Table 2a -- rotation protocol d=2 (arcsine: heavy-tailed Beta law)",
        2, n_rot, np.random.default_rng(args.seed + 2000), cal_rng, args.ncal)
    protocol_table(
        f"Table 2b -- rotation protocol d={d} (Beta((d-1)/2,(d-1)/2): "
        "light-tailed law)",
        d, n_rot, rng, cal_rng, args.ncal)

    print(f"  runtime {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
