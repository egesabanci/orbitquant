"""P3.7 -- ANN scoring over quantized codes with full-precision rerank.

The TurboQuant-style product estimator stores, per database vector, a b-bit
code per coordinate: rotate with a shared orthogonal rotation, then scalar
quantize each rotated coordinate with the exact-Beta Lloyd-Max codebook.
Queries are then scored **directly over the stored codes** in ADC
(asymmetric-distance-computation) style: the code array is scanned blockwise,
each block's code indices resolved through the codebook lookup table, and the
block scores computed as inner products of the rotated query with those
block reconstructions. Only one block of reconstructions is ever in flight --
no full-precision database vectors are touched and no dequantized corpus is
materialized.

Two retrieval pipelines are compared on a synthetic unit-vector dataset
(``n_db`` database vectors, ``n_q`` queries, iid uniform on the sphere):

1. Full dequantization (baseline): dequantize the entire code corpus
   (``D = codebook[codes]``), score with exact inner products against the
   rotated queries, take the top-k by those scores directly.
2. Quantized-score + rerank: score with the ADC code scan, take the top-C
   (C = cand_scale * k) candidates, then rerank only those C in full
   precision -- the original database vectors ``X`` are accessed for the C
   candidates only.

Recall@k (mean overlap of retrieved with the exact ground-truth top-k) is
reported for both pipelines at a few k. Each trial draws an independent Haar
rotation, so the measurement is averaged over independent rotation/quantizer
noise draws per the shared protocol.

Pure NumPy. No SciPy, no model, no GPU. Rotations are ``Rotation`` objects
applied in batch: ``R.forward(X.T).T`` (all rotations in rotations.py support
matrix-shaped input).
"""
from __future__ import annotations

import numpy as np

from . import codebooks as cb
from . import rotations as rot

__all__ = [
    "ann_dataset",
    "product_codes",
    "quantized_scores_scan",
    "full_dequant_scores",
    "recall_at_k",
    "recall_pipelines",
    "ann_recall_trial",
]


def ann_dataset(n_db: int, n_q: int, d: int, rng: np.random.Generator) -> tuple:
    """Synthetic unit-vector dataset: (n_db, d) database and (n_q, d) queries.

    Both are iid uniform on the unit sphere (Gaussian normalized), so inner
    products are near zero-mean with std ~ 1/sqrt(d) -- a well-posed retrieval
    task with a definite ground truth.
    """
    X = rng.standard_normal((n_db, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = rng.standard_normal((n_q, d))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return X, Q


def product_codes(
    X: np.ndarray,
    R: rot.Rotation,
    codebook: np.ndarray,
) -> np.ndarray:
    """b-bit product-estimator codes: rotate, then scalar quantize coordinates.

    Returns an (n, d) integer code array (the stored ANN codes). The shared
    rotation R maps every vector into the coordinate law the codebook is
    trained for (Haar => exact Beta coordinates).
    """
    Y = np.asarray(R.forward(X.T)).T
    return cb.quantize(Y, codebook)


def _rotate_batch(R: rot.Rotation, M: np.ndarray) -> np.ndarray:
    """Apply a Rotation to a (n, d) matrix in batch: returns (n, d)."""
    return np.asarray(R.forward(M.T)).T


def quantized_scores_scan(
    Q: np.ndarray,
    codes: np.ndarray,
    codebook: np.ndarray,
    R: rot.Rotation,
    db_block: int = 1000,
) -> np.ndarray:
    """ADC-style scoring of queries over stored codes (blockwise code scan).

    Scans the stored code array in database blocks: for each block, the code
    indices are resolved through the codebook lookup table (centroid gather
    for that block only) and the block scores are the inner products of the
    rotated queries with those reconstructions:

        score(q, j) = <R q, D(codes[j])>,   D = codebook[.]

    Only one block of reconstructions (``db_block`` rows) is ever in flight,
    so no dequantized corpus is materialized and no full-precision database
    vector is touched -- unlike the ``full_dequant_scores`` baseline, which
    dequantizes the entire corpus in one shot.

    Returns the (n_q, n_db) approximate score matrix.
    """
    Qr = _rotate_batch(R, Q)                      # (n_q, d) rotated queries
    n_q, n_db = Q.shape[0], codes.shape[0]
    S = np.empty((n_q, n_db))
    for lo in range(0, n_db, db_block):
        codes_b = codes[lo : lo + db_block]       # (B, d) int code indices
        D_b = codebook[codes_b]                   # (B, d) block reconstructions
        S[:, lo : lo + D_b.shape[0]] = Qr @ D_b.T
    return S


def full_dequant_scores(
    Q: np.ndarray,
    codes: np.ndarray,
    codebook: np.ndarray,
    R: rot.Rotation,
) -> np.ndarray:
    """Baseline: dequantize the whole code corpus, then exact inner products.

    D = codebook[codes] materializes every database vector's reconstruction
    (the systems path P3.7 replaces); scores = (R Q) @ D.T.
    """
    Qr = _rotate_batch(R, Q)                      # (n_q, d) rotated queries
    D = codebook[codes]                           # (n_db, d) full dequantization
    return Qr @ D.T                               # (n_q, n_db)


def _topk(S: np.ndarray, k: int) -> np.ndarray:
    """Sorted top-k indices per row via argpartition + partial sort (O(n) select)."""
    n_q = S.shape[0]
    if k >= S.shape[1]:
        return np.argsort(-S, axis=1)
    rows = np.arange(n_q)[:, None]
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    v = S[rows, part]
    order = np.argsort(-v, axis=1)
    return part[rows, order]


def recall_at_k(pred_topk: np.ndarray, true_topk: np.ndarray) -> float:
    """Mean recall@k: average of |retrieved ∩ true top-k| / k over queries."""
    k = pred_topk.shape[1]
    match = (pred_topk[..., None] == true_topk[:, None, :]).any(axis=-1)
    return float(match.mean())


def recall_pipelines(
    Q: np.ndarray,
    X: np.ndarray,
    S_baseline: np.ndarray,
    S_scan: np.ndarray,
    true: np.ndarray,
    k: int,
    cand_scale: int,
) -> tuple:
    """Recall@k: (full dequantization, quantized-score + rerank) for one k.

    - full: top-k straight off ``S_baseline``, the scores from dequantizing
      the entire code corpus (the systems path P3.7 replaces).
    - rerank: top-C by ``S_scan``, the ADC scores over the stored codes, then
      exact scores ``<q, x_j>`` computed against the original database vectors
      for the C candidates only.

    ``true`` (exact (n_q, n_db) inner products) is the evaluation oracle used
    to measure recall and to rerank; the rerank path reads back only the C
    candidate columns of ``X``.
    """
    n_q = Q.shape[0]
    true_topk = _topk(true, k)

    full_topk = _topk(S_baseline, k)
    recall_full = recall_at_k(full_topk, true_topk)

    C = cand_scale * k
    cand = _topk(S_scan, C)                       # (n_q, C) candidate set
    rows = np.arange(n_q)[:, None]
    exact_cand = np.einsum("qcd,qd->qc", X[cand], Q)   # <q, x_j> for candidates only
    order_c = np.argsort(-exact_cand, axis=1)[:, :k]
    rerank_topk = cand[rows, order_c]
    recall_rerank = recall_at_k(rerank_topk, true_topk)

    return recall_full, recall_rerank


def ann_recall_trial(
    X: np.ndarray,
    Q: np.ndarray,
    R: rot.Rotation,
    codebook: np.ndarray,
    ks: list,
    cand_scale: int,
    true: np.ndarray,
) -> dict:
    """One-trial recall@k for all k: {k: (full_dequant, score + rerank)}.

    Encodes the corpus once, then scores it both ways -- ADC code scan
    (P3.7 path) and corpus-wide dequantization (baseline) -- and returns the
    two pipelines' recall per k. ``true`` is the cached exact (n_q, n_db)
    ground-truth inner-product matrix, computed once outside the trial loop;
    inside the trial the full-precision database ``X`` is touched only for
    the C rerank candidates (via ``recall_pipelines``).
    """
    codes = product_codes(X, R, codebook)
    S_scan = quantized_scores_scan(Q, codes, codebook, R)   # P3.7: scan over codes
    S_full = full_dequant_scores(Q, codes, codebook, R)     # baseline: dequantize all
    return {
        k: recall_pipelines(Q, X, S_full, S_scan, true, k, cand_scale)
        for k in ks
    }
