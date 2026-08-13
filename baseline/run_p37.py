"""P3.7 -- ANN scoring over quantized codes with full-precision rerank.

Synthetic unit-vector dataset (n_db database vectors, n_q queries), quantized
with the b-bit product estimator (shared Haar rotation + exact-Beta Lloyd-Max
codebook). Queries are scored directly over the quantized codes; the top-C
candidate set is then reranked in full precision. Reports recall@k for full
dequantization (top-k off the approximate scores) vs quantized-score + rerank
at a few k, averaged over nrot independent Haar rotation trials.

Usage:
    python3 -m baseline.run_p37 --d 64 --nrot 2000 [--b 2] [--n_db 5000]
                                [--n_q 100] [--ks 1 5 10] [--cand_scale 8]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import codebooks as cb
from . import p37_ann as ann
from . import rotations as rot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nrot", type=int, default=2000,
                    help="independent Haar rotation trials to average over")
    ap.add_argument("--b", type=int, default=2,
                    help="bits per coordinate of the product estimator")
    ap.add_argument("--n_db", type=int, default=5000,
                    help="database vectors (synthetic, unit norm)")
    ap.add_argument("--n_q", type=int, default=100, help="query vectors")
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 5, 10],
                    help="recall@k values to report")
    ap.add_argument("--cand_scale", type=int, default=8,
                    help="rerank candidate set size C = cand_scale * k")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    d, nrot = args.d, args.nrot
    ks = sorted(args.ks)

    X, Q = ann.ann_dataset(args.n_db, args.n_q, d, rng)
    true = Q @ X.T
    codebook = cb.beta_lloyd_max(d, args.b)
    n_bits = args.n_db * d * args.b  # stored code size

    print(f"P3.7 ANN scoring over quantized codes  d={d} nrot={nrot} b={args.b} "
          f"n_db={args.n_db} n_q={args.n_q} seed={args.seed}")
    print(f"  codebook: exact-Beta Lloyd-Max, {len(codebook)} centroids; "
          f"codes stored = {n_bits/8/1e6:.2f} MB ({args.b} b/coord)")
    print(f"  rerank candidate set C = cand_scale * k = {args.cand_scale} * k")
    print()

    # average recall over nrot independent rotations (fresh quantizer noise)
    acc = {k: {"full": 0.0, "rerank": 0.0} for k in ks}
    t0 = time.perf_counter()
    for i in range(nrot):
        R = rot.rotation_from_name("haar", d, rng)
        for k in ks:
            rf, rr = ann.ann_recall_trial(X, Q, R, codebook, k, args.cand_scale)
            acc[k]["full"] += rf
            acc[k]["rerank"] += rr
    elapsed = time.perf_counter() - t0
    for k in ks:
        acc[k]["full"] /= nrot
        acc[k]["rerank"] /= nrot

    print(f"{'k':>4}{'C':>6}{'full_dequant':>14}{'score+rerank':>14}"
          f"{'gain':>9}")
    print("-" * 47)
    for k in ks:
        f, r = acc[k]["full"], acc[k]["rerank"]
        C = args.cand_scale * k
        print(f"{k:>4}{C:>6}{f:>14.4f}{r:>14.4f}{r - f:>9.4f}")
    print("-" * 47)
    print(f"  recall@k measured against exact ground truth over {nrot} "
          f"rotation trials ({elapsed:.1f}s); ground truth <q,x> mean "
          f"abs = {np.mean(np.abs(true)):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
