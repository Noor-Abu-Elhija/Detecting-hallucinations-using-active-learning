# src/corpus_index.py
from __future__ import annotations
import os, json
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

def _ensure_fp32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=False)

def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    x = _ensure_fp32(x)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def chunk_text(text: str, max_tokens: int = 180, overlap: int = 30) -> List[str]:
    """
    Simple whitespace chunker (token≈word). For SQuAD contexts this works fine.
    """
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
    Stores:
      - doc_texts: list[str] chunks
      - embeddings: np.ndarray (N, D) L2-normalized
      - faiss index: cosine via inner product on normalized vectors
    Save/Load to index_dir: {index.faiss, embs.npy, meta.jsonl}
    """
    def __init__(self, embeddings: np.ndarray, doc_texts: List[str]):
        self.emb = _l2norm_rows(embeddings)
        self.texts = doc_texts
        self.dim = self.emb.shape[1]
        if _HAS_FAISS:
            self.index = faiss.IndexHNSWFlat(self.dim, 32)  # good recall/speed default
            # IndexHNSWFlat expects L2-normalized for cosine via IP if we set metric to IP
            self.index.metric_type = faiss.METRIC_INNER_PRODUCT
            x = self.emb.copy()
            faiss.normalize_L2(x)
            self.index.add(x)
        else:
            self.index = None

    def search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Searches the index for the top-k most similar embeddings.

        Returns:
          (cosine_similarities: np.ndarray, indices: np.ndarray)
        """
        q2 = query_embedding.astype(np.float32).reshape(1, -1)
        # L2-normalize the query vector to prepare for cosine similarity search
        faiss.normalize_L2(q2)

        # Use FAISS to find the indices of the top-k nearest neighbors
        distances, idxs = self.index.search(q2, k)

        # This is the crucial fix:
        # We ignore the returned 'distances' and manually calculate the true
        # cosine similarity (dot product of normalized vectors) for the top-k results.
        # The variable for embeddings in your class is self.emb (singular).
        retrieved_embs = self.emb[idxs[0]]
        sims = (q2 @ retrieved_embs.T).ravel()

        return sims, idxs[0]
    def save(self, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)
        np.save(os.path.join(index_dir, "embs.npy"), self.emb)
        with open(os.path.join(index_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
            for t in self.texts:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(index_dir, "index.faiss"))

    @classmethod
    def load(cls, index_dir: str) -> "CorpusIndex":
        emb = np.load(os.path.join(index_dir, "embs.npy"))
        texts = []
        with open(os.path.join(index_dir, "meta.jsonl"), "r", encoding="utf-8") as f:
            for ln in f:
                texts.append(json.loads(ln)["text"])
        obj = cls(emb, texts)
        # If FAISS file exists, read it (keeps NumPy fallback otherwise)
        if _HAS_FAISS:
            idx_path = os.path.join(index_dir, "index.faiss")
            if os.path.exists(idx_path):
                obj.index = faiss.read_index(idx_path)
        return obj
