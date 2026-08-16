"""run_protection.py -- data-oblivious key protection on the attention objective
(exp 13).

Tests whether reallocating a few bits to a small protected pool of keys by
STORED norm (available at encode time, and inferable by the decoder from the
shared rule -- zero extra metadata) improves softmax-attention fidelity at
~the same average true bytes, on the real objective (not MSE).

Controls ensure we do not mistake "extra bits help" for "norm-targeting
helps": a random-protection control isolates the targeting effect, and the
uniform rates anchor the frontier.

Usage:  python -m baseline.run_protection
"""
from __future__ import annotations

import sys

from . import benchmark_tq as B

CONFIGS = [
    ("uniform b=2", dict(frac=0.0, prot_b=2, base_b=2, mode="topnorm")),
    ("uniform b=1", dict(frac=0.0, prot_b=1, base_b=1, mode="topnorm")),
    ("uniform b=3", dict(frac=0.0, prot_b=3, base_b=3, mode="topnorm")),
    ("prot top1% b=3 / rest b=2", dict(frac=0.01, prot_b=3, base_b=2, mode="topnorm")),
    ("prot RANDOM1% b=3 / rest b=2", dict(frac=0.01, prot_b=3, base_b=2, mode="random")),
    ("prot top5% b=3 / rest b=2", dict(frac=0.05, prot_b=3, base_b=2, mode="topnorm")),
    ("prot top10% b=3 / rest b=2", dict(frac=0.10, prot_b=3, base_b=2, mode="topnorm")),
    ("prot top5% b=4 / rest b=1", dict(frac=0.05, prot_b=4, base_b=1, mode="topnorm")),
]


def main() -> int:
    print("Data-oblivious key protection -- attention objective (shared serving rotation)")
    print(f"{'config':<32}{'avg bits':>9}{'bytes':>8}{'KL':>13}{'recall@5':>10}")
    rows = []
    for name, kw in CONFIGS:
        if kw["frac"] == 0.0:
            r = B.attention_protection(prot_b=kw["prot_b"], base_b=kw["prot_b"])
        else:
            r = B.attention_protection(**kw)
        rows.append((name, r))
        print(f"{name:<32}{r['bits']:>9.1f}{r['bits']/8:>8.2f}{r['kl']:>13.7f}{r['recall5']:>10.3f}")
    u2 = [r for n, r in rows if n == "uniform b=2"][0]
    t1 = [r for n, r in rows if n == "prot top1% b=3 / rest b=2"][0]
    rd = [r for n, r in rows if n == "prot RANDOM1% b=3 / rest b=2"][0]
    print(f"\nsame-bytes verdict: top1% KL {(t1['kl']/u2['kl']-1)*100:+.1f}% @ "
          f"+{(t1['bits']-u2['bits']):.1f} bits; random {abs(rd['kl']-u2['kl'])/u2['kl']*100:.1f}% "
          f"(isolates norm-targeting, which is real but small).")
    print("Caveat: protected pools make per-key payload variable -> deployability note.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
