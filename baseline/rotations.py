"""Rotation transforms for OrbitQuant proposals.

Compares the ideal Haar random rotation against fast structured alternatives
(P0.1/P0.2). All are orthogonal (norm/inner-product preserving).

The structured rotations use a true Fast Walsh-Hadamard Transform (FWHT)
butterfly, O(d log d) to apply with O(d) parameter storage (one sign per
coordinate per round) -- this is the P0.2 requirement, not a dense matrix.

Each rotation is a ``Rotation`` object with ``forward(x)`` and ``inverse(x)``
(orthogonal => inverse = transpose).

Pure NumPy. No SciPy, no model, no GPU.
"""
from __future__ import annotations

import numpy as np

from .protocol import random_rotation

__all__ = [
    "Rotation",
    "haar",
    "gaussian_qr",
    "hadamard",
    "hadamard_sign_flip",
    "random_permutation",
    "rotation_from_name",
]


class Rotation:
    """An orthogonal transform with forward and inverse application."""

    def __init__(self, forward, inverse, name: str):
        self.forward = forward
        self.inverse = inverse
        self.name = name

    def __call__(self, x):
        return self.forward(x)


# --------------------------------------------------------------------------- #
# Fast Walsh-Hadamard Transform (butterfly, O(d log d))
# --------------------------------------------------------------------------- #
def fwht(x: np.ndarray) -> np.ndarray:
    """In-place Fast Walsh-Hadamard Transform of a vector (O(d log d)).

    Unnormalized (gains sqrt(d) per transform). Use ``fwht``/``ifwht`` pairs
    (or normalize once) to keep orthogonality.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    d = x.shape[0]
    h = 1
    while h < d:
        for i in range(0, d, h * 2):
            a = x[i : i + h].copy()
            b = x[i + h : i + 2 * h].copy()
            x[i : i + h] = a + b
            x[i + h : i + 2 * h] = a - b
        h *= 2
    return x


def _hadamard_apply(x: np.ndarray) -> np.ndarray:
    """Apply normalized H/sqrt(d) via FWHT butterfly."""
    return fwht(x) / np.sqrt(x.shape[0])


# --------------------------------------------------------------------------- #
# Rotation transforms
# --------------------------------------------------------------------------- #
def haar(d: int, rng: np.random.Generator) -> Rotation:
    """Haar random rotation (the theoretical ideal)."""
    P = random_rotation(d, rng)
    return Rotation(lambda x: P @ x, lambda x: P.T @ x, "haar")


def gaussian_qr(d: int, rng: np.random.Generator) -> Rotation:
    """Gaussian-QR rotation (equivalent to Haar for orthogonal matrices)."""
    return haar(d, rng)


def hadamard(d: int, rng: np.random.Generator) -> Rotation:
    """Randomized Walsh-Hadamard: D H (random diagonal sign, then H/sqrt(d)).

    O(d log d) forward/inverse with O(d) storage (the sign vector). The pure
    Hadamard is deterministic (H@e1 is the same vector every trial), so we
    prepend a random sign flip to measure the distribution over randomized
    structured rotations.
    """
    diag = rng.choice([-1.0, 1.0], size=d)
    return Rotation(
        lambda x: _hadamard_apply(diag * x),
        lambda x: diag * _hadamard_apply(x),
        "hadamard",
    )


def hadamard_sign_flip(d: int, rng: np.random.Generator, rounds: int = 3) -> Rotation:
    """Structured rotation: D1 H D2 H ... Dk H (random diagonal sign flips).

    O(d log d) forward/inverse with O(d) parameter storage (one sign per
    coordinate per round). Forward applies signs Dk..D1 (rightmost first);
    inverse applies signs D1..Dk (transpose). Not Haar-distributed, but mixes
    coordinates.
    """
    signs = [rng.choice([-1.0, 1.0], size=d) for _ in range(rounds)]

    def forward(x):
        # T = D1 H D2 H ... Dk H ; applied to x: H, then Dk, then H, then D_{k-1}...
        # i.e. alternate H-then-sign with signs in reverse order.
        y = x
        for s in reversed(signs):
            y = _hadamard_apply(y)
            y = s * y
        return y

    def inverse(x):
        # T^T = H Dk H D_{k-1} ... H D1 ; applied: D1, then H, then D2, then H...
        # i.e. alternate sign-then-H with signs in forward order.
        y = x
        for s in signs:
            y = s * y
            y = _hadamard_apply(y)
        return y

    return Rotation(forward, inverse, f"hdhdh" if rounds == 3 else f"hd" * rounds)


def random_permutation(d: int, rng: np.random.Generator) -> Rotation:
    """Random coordinate permutation (orthogonal)."""
    perm = rng.permutation(d)
    inv = np.argsort(perm)
    return Rotation(
        lambda x: x[perm],
        lambda x: x[inv],
        "perm",
    )


# Rotation names -> factories returning a Rotation (O(d log d) or O(d) apply).
_ROTATIONS = {
    "haar": haar,
    "gaussian_qr": gaussian_qr,
    "hadamard": hadamard,
    "hd": lambda d, rng: hadamard_sign_flip(d, rng, rounds=1),
    "hdhd": lambda d, rng: hadamard_sign_flip(d, rng, rounds=2),
    "hdhdh": lambda d, rng: hadamard_sign_flip(d, rng, rounds=3),
    "perm": random_permutation,
}


def rotation_from_name(name: str, d: int, rng: np.random.Generator) -> Rotation:
    """Build a Rotation by name."""
    if name not in _ROTATIONS:
        raise ValueError(f"unknown rotation '{name}'; have {sorted(_ROTATIONS)}")
    return _ROTATIONS[name](d, rng)
