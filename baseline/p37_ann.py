"""P3.7 -- ANN scoring over quantized codes with full-precision rerank.

The TurboQuant-style product estimator stores, per database vector, a b-bit
code per coordinate: rotate with a shared orthogonal rotation, then scalar
quantize each rotated coordinate with the exact-Beta Lloyd-Max codebook. A
query is then scored directly over the quantized codes via the product
estimator  <q, Q(z)> = <R q, D(R z)>,  i.e. an exact inner product between the
rotated query and the dequantized database codes -- no full-precision vectors
are touched during the scan.

Two retrieval pipelines are compared on a synthetic unit-vector dataset
(``n_db`` database vectors, ``n_q`` queries, iid uniform on the sphere):

1. Full dequantization: dequantize every database code, take the top-k by the
   approximate scores directly.
2. Quantized-score + rerank: take the top-C (C = cand_scale * k) candidates
   by the approximate scores, then rerank only those C in full precision
   (exact inner products against the original vectors).

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
    "quantized_scores",
    "recall_at_k",
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


def quantized_scores(
    Q: np.ndarray,
    codes: np.ndarray,
    codebook: np.ndarray,
    R: rot.Rotation,
) -> np.ndarray:
    """Approximate inner products <q, Q(z)> via the product estimator.

    Returns (n_q, n_db) scores = (R Q) @ D(codes)^T, computed entirely over
    the quantized codes (D(codes) = dequantized centroids); no full-precision
    database vectors are touched.
    """
    Qr = np.asarray(R.forward(Q.T)).T  # (n_q, d) rotated queries
    D = codebook[codes]                # (n_db, d) dequantized database codes
    return Qr @ D.T                    # (n_q, n_db)


def recall_at_k(pred_topk: np.ndarray, true_topk: np.ndarray) -> float:
    """Mean recall@k: average of |retrieved ∩ true top-k| / k over queries."""
    k = pred_topk.shape[1]
    hits = 0.0
    for q in range(pred_topk.shape[0]):
        hits += len(np.intersect1d(pred_topk[q], true_topk[q]))
    return float(hits / pred_topk.shape[0] / k)


def ann_recall_trial(
    X: np.ndarray,
    Q: np.ndarray,
    R: rot.Rotation,
    codebook: np.ndarray,
    k: int,
    cand_scale: int,
) -> tuple:
    """One-trial recall@k for one k: (full_dequant, quantized_score + rerank).

    ``true`` is the exact (n_q, n_db) inner-product matrix; the rerank step
    reads back only the top-C candidate columns from it, mimicking a
    full-precision rerank over a small candidate set.
    """
    n_q = Q.shape[0]
    true = Q @ X.T                      # (n_q, n_db) exact inner products
    true_topk = np.argsort(-true, axis=1)[:, :k]

    codes = product_codes(X, R, codebook)
    S = quantized_scores(Q, codes, codebook, R)

    # 1. full dequantization: top-k straight off the approximate scores
    full_topk = np.argsort(-S, axis=1)[:, :k]
    recall_full = recall_at_k(full_topk, true_topk)

    # 2. quantized-score + rerank: top-C by approximate scores, then exact
    C = cand_scale * k
    cand = np.argsort(-S, axis=1)[:, :C]
    rows = np.arange(n_q)[:, None]
    exact_cand = true[rows, cand]       # exact scores for candidates only
    order = np.argsort(-exact_cand, axis=1)[:, :k]
    rerank_topk = cand[rows, order]
    recall_rerank = recall_at_k(rerank_topk, true_topk)

    return recall_full, recall_rerank
