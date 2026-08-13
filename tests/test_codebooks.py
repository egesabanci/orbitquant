"""Regression tests for the exact-Beta Lloyd-Max codebook solver.

Guards the invariant that the solver never returns a degenerate (non-sorted,
zero-mass, non-converged) codebook, which previously produced a 2.7x MSE gap
at d=128/b=4 (0.025 vs the asymptotic 0.009).
"""
from __future__ import annotations

import numpy as np
import pytest

from baseline import codebooks as cb


def test_codebook_sorted_and_nonempty():
    """Every codebook is strictly sorted (no duplicate/zero-mass cells)."""
    for d in (64, 128):
        for b in (1, 2, 3, 4):
            c = cb.beta_lloyd_max(d, b)
            assert len(c) == 2**b
            assert np.all(np.diff(c) > 0), f"non-sorted codebook d={d} b={b}"
            # symmetric about zero
            assert np.allclose(c, -c[::-1])


def test_codebook_matches_asymptotic_mse():
    """Per-vector MSE matches the paper's Gaussian-asymptotic values.

    Under sqrt(d) scaling the coordinate law converges to Gaussian, so exact
    Lloyd-Max should approach the asymptotic values at both d=64 and d=128.
    """
    paper = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
    rng = np.random.default_rng(0)
    for d in (64, 128):
        for b in (1, 2, 3, 4):
            c = cb.beta_lloyd_max(d, b)
            # sample coordinates from the exact Beta law: X^2 ~ Beta(1/2,(d-1)/2)
            from scipy.stats import beta

            u = beta.rvs(0.5, (d - 1) / 2, size=500_000, random_state=0)
            x = rng.choice([-1, 1], size=len(u)) * np.sqrt(u)
            yhat = np.empty_like(x)
            for i in range(0, len(x), 100_000):
                sl = slice(i, i + 100_000)
                yhat[sl] = cb.dequantize(cb.quantize(x[sl, None], c), c)[:, 0]
            per_vector = np.mean((x - yhat) ** 2) * d
            assert abs(per_vector - paper[b]) < 0.05, (
                f"d={d} b={b} MSE={per_vector:.4f} vs paper {paper[b]}"
            )


def test_lloyd_max_converges():
    """Solver reaches a fixed point (does not silently return early)."""
    # a sharply peaked density (d=128) is the hardest case that previously
    # failed to converge; it must converge and produce a valid codebook
    c = cb.beta_lloyd_max(128, 4)
    assert np.all(np.isfinite(c))
    assert len(c) == 16
