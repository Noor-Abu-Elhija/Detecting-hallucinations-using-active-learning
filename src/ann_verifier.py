# src/ann_verifier.py
# This module implements a cosine-similarity-based Approximate Nearest Neighbor (ANN) verifier.
# It uses FAISS for efficient search if available, otherwise falls back to a NumPy implementation.

from __future__ import annotations
import numpy as np

try:
    import faiss  # pip install faiss-cpu
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization for 1D or 2D NumPy arrays."""
    x = x.astype(np.float32, copy=False)
    if x.ndim == 1:
        n = np.linalg.norm(x) + 1e-12
        return x / n
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


class ANNVerifier:
    """
    Cosine-similarity ANN verifier for factual support detection.

    - Uses FAISS (IndexFlatIP) when available for fast inner-product search.
    - Falls back to NumPy cosine similarity if FAISS is unavailable.
    """

    def __init__(self, embeddings: np.ndarray):
        """
        Initialize the verifier with a trusted corpus of embeddings.
        Args:
            embeddings: (N, D) array of float32 embeddings representing reference texts.
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self._emb = _l2_normalize(embeddings)
        self.dim = self._emb.shape[1]

        if _HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dim)
            emb_norm = self._emb.copy()
            faiss.normalize_L2(emb_norm)
            self.index.add(emb_norm)
        else:
            self.index = None  # fallback mode

    def verify(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.80):
        """
        Compare a query embedding against the corpus.
        Returns:
            (is_supported, max_cosine_sim, idxs)
        where:
            - is_supported: True if top cosine similarity >= threshold
            - max_cosine_sim: top similarity score
            - idxs: indices of the top-k similar entries
        """
        q = _l2_normalize(query_embedding.astype(np.float32).reshape(1, -1))

        if self.index is not None:  # FAISS-based search
            q_f = q.copy()
            faiss.normalize_L2(q_f)
            sims, idxs = self.index.search(q_f, k)
            sims, idxs = sims[0], idxs[0]
        else:  # NumPy fallback
            sims_full = (q @ self._emb.T).ravel()
            if k >= sims_full.size:
                topk_idx = np.argsort(-sims_full)
            else:
                part = np.argpartition(-sims_full, k)[:k]
                topk_idx = part[np.argsort(-sims_full[part])]
            sims, idxs = sims_full[topk_idx], topk_idx

        max_sim = float(sims[0]) if sims.size else 0.0
        is_supported = max_sim >= threshold
        return is_supported, max_sim, idxs
