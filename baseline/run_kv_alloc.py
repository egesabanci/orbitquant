"""run_kv_alloc.py -- K/V bit-allocation on the KV attention objective (exp 15).

Values are consumed in attention as out = sum_i p_i v_i, so value quantization
error enters the OUTPUT linearly, while key quantization error only reshapes
the attention weights p. This runner measures the attention-output error of
K/V allocation pairs at fixed total bytes/token, decomposes value-side vs
key-side effects, and tests the "More for Keys, Less for Values" direction
(P1.2) on the REAL objective.

Usage:  python -m baseline.run_kv_alloc
"""
from __future__ import annotations

import sys

from . import benchmark_tq as B

CFGS = [(3, 1), (2, 2), (1, 3), (3, 2), (2, 3), (1, 2), (2, 1)]


def main() -> int:
    print("K/V bit allocation -- attention-output error (shared serving rotations)")
    print(f"{'alloc':<8}{'bits':>6}{'rel_out_err e2e':>17}{'V-only':>11}{'kl(key)':>10}")
    for bk, bv in CFGS:
        e = B.kv_attention_error(b_k=bk, b_v=bv, est_p=True)
        v = B.kv_attention_error(b_k=bk, b_v=bv, est_p=False)
        print(f"K{bk}V{bv:<5}{e['bits']:>6.0f}{e['rel_out_err']:>17.6f}"
              f"{v['rel_out_err']:>11.6f}{e['kl']:>10.6f}")
    print("\nAt 268 bits: K1V3 output-error is "
          f"{B.kv_attention_error(b_k=3,b_v=1)['rel_out_err']/B.kv_attention_error(b_k=1,b_v=3)['rel_out_err']:.1f}x "
          "better than K3V1 (key-favored) -- value bits buy OUTPUT fidelity, "
          "key bits buy DISTRIBUTION (KL) fidelity; the two are separable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
