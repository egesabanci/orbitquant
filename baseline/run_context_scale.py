"""run_context_scale.py -- attention-objective fidelity vs context length (exp 16).

KV compression exists for long context; every prior probe used one key-set
size. This sweeps the served context length (same rotation, same queries, same
pipeline) and reports softmax-KL, recall@5, p95-logit ratio and per-key
score-noise, for b=1 and b=2, plus the fast hdhdh rotation at 8k keys.

Usage:  python -m baseline.run_context_scale
"""
from __future__ import annotations

import sys

from . import benchmark_tq as B


def main() -> int:
    print("Attention-objective fidelity vs context length (d=64, same "
          "rotation/queries):")
    print(f"{'n_db':>6} | {'KL b=1':>10} {'KL b=2':>10} | {'r5 b=1':>7} "
          f"{'r5 b=2':>7} | {'noise b=2':>9} | {'p95 b=2':>8}")
    r1 = B.attention_context_scale(b=1)
    r2 = B.attention_context_scale(b=2)
    for a, c in zip(r1, r2):
        print(f"{a['n_db']:>6} | {a['kl']:>10.6f} {c['kl']:>10.6f} | "
              f"{a['recall5']:>7.3f} {c['recall5']:>7.3f} | "
              f"{c['score_noise']:>9.4f} | {c['p95_logit_ratio']:>8.3f}")
    m8 = B.attention_metrics(64, 2, n_db=8000, rot="hdhdh")
    print(f"\nhdhdh @ n_db=8000: KL {m8['kl']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
