# src/corpus_index.py
# This module builds, saves, and searches a corpus index for text embeddings.
# It supports FAISS-based Approximate Nearest Neighbor (ANN) search or NumPy fallback.

from __future__ import annotations
import os, json
from typing import List
import numpy as np

try:
    import faiss  # pip install faiss-cpu
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False


def _ensure_fp32(x: np.ndarray) -> np.ndarray:
    """Ensure array is in float32 format for FAISS compatibility."""
    return x.astype(np.float32, copy=False)


def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization."""
    x = _ensure_fp32(x)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def chunk_text(text: str, max_tokens: int = 180, overlap: int = 30) -> List[str]:
    """Simple whitespace-based chunker (token≈word), suitable for SQuAD contexts."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_tokens]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += max_tokens - overlap
    return chunks


class CorpusIndex:
    """
    CorpusIndex stores a set of document embeddings and their texts.
    It supports cosine similarity search using FAISS or NumPy.
    """

    def __init__(self, embeddings: np.ndarray, doc_texts: List[str]):
        """
        Args:
            embeddings: np.ndarray (N, D) — document embeddings.
            doc_texts: list[str] — corresponding text chunks.
        """
        self.emb = _l2norm_rows(embeddings)
        self.texts = doc_texts
        self.dim = self.emb.shape[1]

        if _HAS_FAISS:
            self.index = faiss.IndexHNSWFlat(self.dim, 32)
            self.index.metric_type = faiss.METRIC_INNER_PRODUCT
            x = self.emb.copy()
            faiss.normalize_L2(x)
            self.index.add(x)
        else:
            self.index = None

    def search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Retrieve top-k most similar embeddings by cosine similarity.

        Returns:
            sims: np.ndarray of cosine similarities.
            idxs: np.ndarray of retrieved indices.
        """
        q2 = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q2)
        distances, idxs = self.index.search(q2, k)

        retrieved_embs = self.emb[idxs[0]]
        sims = (q2 @ retrieved_embs.T).ravel()
        return sims, idxs[0]

    def save(self, index_dir: str):
        """Save embeddings, metadata, and FAISS index (if available)."""
        os.makedirs(index_dir, exist_ok=True)
        np.save(os.path.join(index_dir, "embs.npy"), self.emb)

        with open(os.path.join(index_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
            for t in self.texts:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

        if self.index is not None:
            faiss.write_index(self.index, os.path.join(index_dir, "index.faiss"))

    @classmethod
    def load(cls, index_dir: str) -> "CorpusIndex":
        """Load a previously saved index from disk."""
        emb = np.load(os.path.join(index_dir, "embs.npy"))
        texts = []
        with open(os.path.join(index_dir, "meta.jsonl"), "r", encoding="utf-8") as f:
            for ln in f:
                texts.append(json.loads(ln)["text"])

        obj = cls(emb, texts)

        if _HAS_FAISS:
            idx_path = os.path.join(index_dir, "index.faiss")
            if os.path.exists(idx_path):
                obj.index = faiss.read_index(idx_path)

        return obj
