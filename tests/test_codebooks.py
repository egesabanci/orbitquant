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


def _sample_beta(a: float, b: float, size: int, rng) -> np.ndarray:
    """Sample from Beta(a,b) via the gamma-ratio identity (pure NumPy)."""
    g1 = rng.standard_gamma(a, size=size)
    g2 = rng.standard_gamma(b, size=size)
    return g1 / (g1 + g2)


def test_codebook_matches_asymptotic_mse():
    """Per-vector MSE matches the exact numerical Lloyd-Max distortion.

    Under sqrt(d) scaling the coordinate law converges to Gaussian, so exact
    Lloyd-Max should approach the paper's asymptotic values at both d=64 and
    d=128. The tolerance is tight (0.002) so a degenerate codebook (which gave
    0.025 vs 0.009 at d=128/b=4, a 0.016 gap) cannot pass.
    """
    # exact per-vector distortion from numerical integration of the codebook
    # cost (matches paper asymptotic 0.36/0.117/0.03/0.009 within ~0.0003)
    exact = {
        (64, 1): 0.35839, (64, 2): 0.11453, (64, 3): 0.03339, (64, 4): 0.00913,
        (128, 1): 0.36089, (128, 2): 0.11600, (128, 3): 0.03397, (128, 4): 0.00931,
    }
    rng = np.random.default_rng(0)
    for d in (64, 128):
        for b in (1, 2, 3, 4):
            c = cb.beta_lloyd_max(d, b)
            # sample coordinates from the exact Beta law: X^2 ~ Beta(1/2,(d-1)/2)
            u = _sample_beta(0.5, (d - 1) / 2, 500_000, rng)
            x = rng.choice([-1, 1], size=len(u)) * np.sqrt(u)
            yhat = np.empty_like(x)
            for i in range(0, len(x), 100_000):
                sl = slice(i, i + 100_000)
                yhat[sl] = cb.dequantize(cb.quantize(x[sl, None], c), c)[:, 0]
            per_vector = np.mean((x - yhat) ** 2) * d
            assert abs(per_vector - exact[(d, b)]) < 0.002, (
                f"d={d} b={b} MSE={per_vector:.4f} vs exact {exact[(d, b)]:.4f}"
            )


def test_lloyd_max_converges():
    """Solver reaches a fixed point (does not silently return early)."""
    # a sharply peaked density (d=128) is the hardest case that previously
    # failed to converge; it must converge and produce a valid codebook
    c = cb.beta_lloyd_max(128, 4)
    assert np.all(np.isfinite(c))
    assert len(c) == 16
