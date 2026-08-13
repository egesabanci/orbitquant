"""P2.2 -- Structured residual QJL runner.

Replaces the dense Gaussian residual sketch of the b-bit product estimator
(TurboQuant Algorithm 2 / P1.2 form) with a structured SRHT/FWHT sketch
(random sign + Hadamard, O(d log d) via the FWHT butterfly) and reports
measured bias and variance of the dense vs structured sketch at the same
sketch dimension m, under the standard protocol (fixed x = e1, fixed y with
nonzero dot, independent rotation+sketch per trial).

Table 1: pure QJL product estimator (b_mse = 0, sketch applied to x = e1)
  -- both sketches are exactly unbiased here; the structured sketch is
  deterministic (zero variance) at m = d.
Table 2: full residual product estimator (b_mse-bit MSE base + residual
  sketch) -- dense stays exactly unbiased for any residual; the structured
  sketch keeps a small measured bias (c(r) - 1)<y, r> since c(r) = E|s . r| /
  ||r||^2 ~= 1 for the quantization residual.

Usage:
    python3 -m baseline.run_p22 --d 64 --nrot 2000 [--b_mse 3] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import codebooks as cb
from . import p22_structured_qjl as p22
from . import protocol as pr


def _table(x, y, b_mse, m_values, codebooks, n_rot, rng, header):
    """Print dense vs structured bias/variance rows for one estimator."""
    print(header)
    print(f"{'m':>6}{'dense_bias':>13}{'dense_var':>13}"
          f"{'struct_bias':>14}{'struct_var':>13}{'var_ratio':>11}")
    print("-" * 70)
    for m in m_values:
        dbias, dvar = p22.product_stats(x, y, b_mse, m, codebooks, n_rot, rng, "dense")
        sbias, svar = p22.product_stats(x, y, b_mse, m, codebooks, n_rot, rng, "structured")
        ratio = svar / dvar if dvar > 0 else float("nan")
        print(f"{m:>6}{dbias:>13.5f}{dvar:>13.5f}"
              f"{sbias:>14.5f}{svar:>13.5f}{ratio:>11.3f}")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000)
    ap.add_argument("--b_mse", type=int, default=3,
                    help="scalar MSE base bits for the residual stage (default 3)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d, n_rot = args.d, args.nrot
    rng = np.random.default_rng(args.seed)
    x = pr.fixed_vector(d)
    y = pr.fixed_vector(d, kind="all_equal")
    true = float(np.dot(y, x))

    print(f"P2.2 structured residual QJL  d={d} nrot={n_rot} b_mse={args.b_mse} "
          f"seed={args.seed}")
    print(f"  true <y,x> = {true:.4f}   (x = e1, y = all_equal)")
    print(f"  dense:      S ~ N(0,I) rows,      const sqrt(pi/2)")
    print(f"  structured: SRHT rows (1/sqrt(d)) H D, const sqrt(d)")
    print()

    m_values = [max(1, d // 4), max(1, d // 2), d]
    if args.b_mse > 0:
        codebooks = {args.b_mse: cb.beta_lloyd_max(d, args.b_mse)}
    else:
        codebooks = {}

    _table(
        x, y, 0, m_values, codebooks, n_rot, rng,
        "Table 1 -- pure QJL product estimator (b_mse=0, sketch of x=e1):",
    )
    if args.b_mse > 0:
        _table(
            x, y, args.b_mse, m_values, codebooks, n_rot, rng,
            f"Table 2 -- full residual product estimator "
            f"(b_mse={args.b_mse}-bit base + residual sketch):",
        )

    print("bias: E[est] - <y,x> (unbiased ~0);  var_ratio: structured/dense "
          "variance at same m")
    print("exact unbiasedness of the structured sketch holds only for the "
          "pure-QJL path (r = x = e1, c(r)=1);")
    print("on the residual path (b_mse>0) it keeps a small measured bias "
          "(c(r)-1)<y,r>, ~1e-3 here, since c(r)=E|s.r|/||r||^2 ~= 1 "
          "for a spread residual")
    return 0


if __name__ == "__main__":
    sys.exit(main())
