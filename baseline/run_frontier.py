"""run_frontier.py -- attention-objective rate-distortion frontier (exp 12).

For each honest representation reports (true bits/token, softmax-KL, recall@5)
over a realistic key+query set under one shared serving rotation, plus the
impact of the design choices on the REAL objective:
  - scalar b={1,2,3,4} (6-bit norm header)          -- the rate curve
  - scalar b=2 with 8-bit norm                      -- norm-bits sensitivity
  - prod b=2 (1-bit MSE base + m=64 QJL, 140 bits)  -- the paper's prod path

Usage:  python -m baseline.run_frontier [--d 64]
"""
from __future__ import annotations

import argparse
import sys

from . import benchmark_tq as B


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    args = ap.parse_args(argv)

    rows = B.attention_frontier(d=args.d)
    print(f"Attention-objective rate-distortion frontier  d={args.d} "
          f"(single shared serving rotation; softmax-KL, recall@5)")
    print(f"{'config':<26}{'bits/tok':>9}{'bytes/tok':>10}{'KL':>13}{'recall@5':>10}")
    for r in rows:
        print(f"{r['label']:<26}{r['bits']:>9.0f}{r['bits']/8:>10.2f}"
              f"{r['kl']:>13.7f}{r['recall5']:>10.3f}")

    b2 = [r for r in rows if r["label"] == "scalar b=2"]
    p2 = [r for r in rows if r["label"].startswith("prod")]
    if b2 and p2:
        print(f"\nprod b=2 vs scalar b=2: KL {p2[0]['kl']/b2[0]['kl']:.2f}x worse, "
              f"recall@5 {(1 - p2[0]['recall5'])/(1 - b2[0]['recall5']):.2f}x error")
    scalar = [r for r in rows if r["label"].startswith("scalar b") and "8-bit" not in r["label"]]
    if len(scalar) >= 2:
        k1, k2 = scalar[0]["kl"], scalar[1]["kl"]
        print(f"KL per additional bit/coord (b=1->2): {k1/k2:.2f}x reduction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
