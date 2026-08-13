"""P3.5 -- Block-structured hybrid rotation.

A fast structured transform (randomized FWHT: D H D H D H, the ``hdhdh``
rotation) followed by a small dense mixing layer: block-diagonal orthogonal
matrices acting on contiguous blocks of ``block`` coordinates (e.g. 8x8 or
16x16).

Rationale: the pure FWHT butterfly mixes coordinates globally in O(d log d),
but with very restricted structure -- each output coordinate is a signed sum
of a specific subset of inputs, so residual local structure survives (fixed
e1 produces the "all equal" pattern after one H). The dense layer adds
locally fully-mixed orthogonal mixing inside each block, which the butterfly
never provides, at O(d * block) cost.

Each rotation is a ``Rotation`` object with ``forward(x)`` and ``inverse(x)``
(orthogonal => inverse = transpose). Pure NumPy; no SciPy, no model, no GPU.
"""
from __future__ import annotations

import numpy as np

from . import rotations as rot
from .protocol import random_rotation

__all__ = ["block_mixing_rotation", "orthogonal_blocks"]


def orthogonal_blocks(d: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """(n_blocks, block, block) random orthogonal matrices, one per block.

    Each block is a Haar-distributed orthogonal matrix (sign-corrected QR of a
    Gaussian), so the assembled block-diagonal matrix is orthogonal.
    """
    if d % block != 0:
        raise ValueError(f"d={d} must be divisible by block={block}")
    n_blocks = d // block
    B = np.empty((n_blocks, block, block))
    for b in range(n_blocks):
        B[b] = random_rotation(block, rng)
    return B


def _mix(B: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply block-diagonal mixing: each contiguous block mixed by B[b]."""
    y = np.asarray(x, dtype=np.float64).reshape(B.shape[0], B.shape[1])
    return np.einsum("bij,bj->bi", B, y).reshape(-1)


def _unmix(B: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply transpose of block-diagonal mixing (orthogonal => inverse)."""
    y = np.asarray(x, dtype=np.float64).reshape(B.shape[0], B.shape[1])
    return np.einsum("bji,bj->bi", B, y).reshape(-1)


def block_mixing_rotation(
    d: int,
    rng: np.random.Generator,
    block: int = 8,
    rounds: int = 3,
) -> rot.Rotation:
    """Hybrid rotation: fast FWHT rounds (hdhdh) then a dense mixing layer.

    T = M_block @ (D1 H D2 H ... Dk H)

    where M_block is a block-diagonal orthogonal matrix with ``block``-sized
    random orthogonal blocks (default 8x8; 16 also common). ``rounds`` is the
    number of sign-flip/Hadamard rounds in the fast part (3 => hdhdh).

    O(d log d + d * block) forward/inverse with O(d + d * block) storage.
    """
    base = rot.hadamard_sign_flip(d, rng, rounds=rounds)
    B = orthogonal_blocks(d, block, rng)
    return rot.Rotation(
        lambda x: _mix(B, base.forward(x)),
        lambda x: base.inverse(_unmix(B, x)),
        f"hybrid-b{block}",
    )
