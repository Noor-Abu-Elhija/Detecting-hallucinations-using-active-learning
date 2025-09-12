# src/ann_verifier.py
from __future__ import annotations
import numpy as np

try:
    import faiss  # pip install faiss-cpu
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalize (works for both 1D and 2D)."""
    x = x.astype(np.float32, copy=False)
    if x.ndim == 1:
        n = np.linalg.norm(x) + 1e-12
        return x / n
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


class ANNVerifier:
    """
    Cosine-similarity ANN verifier.

    - If FAISS is available: uses IndexFlatIP (inner product). With normalized vectors,
      inner product == cosine similarity.
    - If FAISS is NOT available: falls back to NumPy cosine search.
    """

    def __init__(self, embeddings: np.ndarray):
        """
        embeddings: (N, D) float32. These are the trusted corpus (e.g., Wikipedia chunks).
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self._emb = _l2_normalize(embeddings)  # keep normalized copy for fallback
        self.dim = self._emb.shape[1]

        if _HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dim)
            # Normalize *again* for FAISS (no-op if already normalized but safe)
            emb_norm = self._emb.copy()
            faiss.normalize_L2(emb_norm)
            self.index.add(emb_norm)
        else:
            self.index = None  # NumPy fallback

    def verify(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.80):
        """
        Returns:
          (is_supported: bool, max_cosine_sim: float, idxs: np.ndarray[int])

        threshold is cosine similarity in [0,1]. Typical values: 0.75–0.90.
        """
        q = query_embedding.astype(np.float32).reshape(1, -1)
        q = _l2_normalize(q)

        if self.index is not None:  # FAISS path
            q_f = q.copy()
            faiss.normalize_L2(q_f)
            sims, idxs = self.index.search(q_f, k)
            sims = sims[0]        # shape (k,)
            idxs = idxs[0]        # shape (k,)
        else:  # NumPy fallback
            # cosine = dot(q, emb) since both are L2-normalized
            sims_full = (q @ self._emb.T).ravel()  # shape (N,)
            # top-k via argpartition
            if k >= sims_full.size:
                topk_idx = np.argsort(-sims_full)
            else:
                part = np.argpartition(-sims_full, k)[:k]
                topk_idx = part[np.argsort(-sims_full[part])]
            sims = sims_full[topk_idx]
            idxs = topk_idx

        max_sim = float(sims[0]) if sims.size else 0.0
        is_supported = (max_sim >= threshold)
        return is_supported, max_sim, idxs
