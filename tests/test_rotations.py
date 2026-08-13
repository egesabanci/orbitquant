"""Tests for the rotation transforms (P0.1/P0.2)."""
from __future__ import annotations

import numpy as np
import pytest

from baseline import rotations as rot


def test_fwht_matches_matrix():
    """FWHT butterfly matches the explicit Walsh-Hadamard matrix."""
    H4 = np.array(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
        dtype=float,
    )
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(rot.fwht(x), H4 @ x)


def test_rotations_roundtrip():
    """Every rotation is orthogonal: inverse(forward(x)) == x."""
    rng = np.random.default_rng(0)
    d = 64
    for name in ["haar", "hadamard", "hd", "hdhd", "hdhdh", "perm"]:
        R = rot.rotation_from_name(name, d, rng)
        x = rng.standard_normal(d)
        y = R.forward(x)
        xr = R.inverse(y)
        assert np.allclose(xr, x, atol=1e-10), f"{name} roundtrip failed"
        assert np.allclose(np.linalg.norm(y), np.linalg.norm(x)), f"{name} not orthogonal"


def test_hdhdh_matches_haar_distribution():
    """hdhdh coordinates follow the Beta law (P0.2 core finding)."""
    from baseline import codebooks as cb
    from baseline import protocol as pr

    rng = np.random.default_rng(0)
    d = 64
    x = pr.fixed_vector(d)
    samples = np.empty(2000 * d)
    for i in range(2000):
        R = rot.rotation_from_name("hdhdh", d, rng)
        samples[i * d : (i + 1) * d] = R.forward(x)
    ks = pr.beta_ks(d, samples)
    assert ks < 0.05, f"hdhdh beta_ks={ks:.4f} too high (not Beta-distributed)"
