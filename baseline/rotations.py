"""Rotation transforms for OrbitQuant proposals.

Compares the ideal Haar random rotation against fast structured alternatives
(P0.1/P0.2). All are orthogonal (norm/inner-product preserving).

Pure NumPy. No SciPy, no model, no GPU.
"""
from __future__ import annotations

import numpy as np

from .protocol import random_rotation

__all__ = [
    "haar",
    "gaussian_qr",
    "hadamard",
    "hadamard_sign_flip",
    "random_permutation",
    "rotation_from_name",
]


def haar(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar random rotation (the theoretical ideal)."""
    return random_rotation(d, rng)


def gaussian_qr(d: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian-QR rotation (equivalent to Haar for orthogonal matrices)."""
    return random_rotation(d, rng)


def _hadamard_matrix(d: int) -> np.ndarray:
    """Sylvester-constructed Walsh-Hadamard matrix of order d (power of 2)."""
    assert d & (d - 1) == 0, "d must be a power of 2"
    h = np.array([[1.0]])
    while h.shape[0] < d:
        h = np.block([[h, h], [h, -h]])
    return h


def hadamard(d: int, rng: np.random.Generator) -> np.ndarray:
    """Randomized Walsh-Hadamard: D H (random diagonal sign, then H/sqrt(d)).

    The pure Hadamard is deterministic (H@e1 is the same vector every trial),
    so to measure the distribution over randomized structured rotations we
    prepend a random sign flip. Orthogonal, O(d log d).
    """
    h = _hadamard_matrix(d) / np.sqrt(d)
    diag = rng.choice([-1.0, 1.0], size=d)
    return h @ (diag[:, None] * np.eye(d))


def hadamard_sign_flip(
    d: int, rng: np.random.Generator, rounds: int = 3
) -> np.ndarray:
    """Structured rotation: D1 H D2 H ... Dk H (random diagonal sign flips).

    O(d log d) compute, tiny parameter storage (one sign per coordinate per
    round), GPU-friendly butterfly structure. Not Haar-distributed, but mixes
    coordinates. ``rounds`` = number of (H then D) stages.
    """
    h = _hadamard_matrix(d) / np.sqrt(d)
    result = np.eye(d)
    for _ in range(rounds):
        diag = rng.choice([-1.0, 1.0], size=d)
        result = h @ (diag[:, None] * result)
    return result


def random_permutation(d: int, rng: np.random.Generator) -> np.ndarray:
    """Random coordinate permutation (orthogonal)."""
    perm = rng.permutation(d)
    P = np.zeros((d, d))
    P[np.arange(d), perm] = 1.0
    return P


_ROTATIONS = {
    "haar": haar,
    "gaussian_qr": gaussian_qr,
    "hadamard": hadamard,
    "hd": lambda d, rng: hadamard_sign_flip(d, rng, rounds=1),
    "hdhd": lambda d, rng: hadamard_sign_flip(d, rng, rounds=2),
    "hdhdh": lambda d, rng: hadamard_sign_flip(d, rng, rounds=3),
    "perm": random_permutation,
}


def rotation_from_name(name: str, d: int, rng: np.random.Generator) -> np.ndarray:
    """Build a rotation matrix by name."""
    if name not in _ROTATIONS:
        raise ValueError(f"unknown rotation '{name}'; have {sorted(_ROTATIONS)}")
    return _ROTATIONS[name](d, rng)
