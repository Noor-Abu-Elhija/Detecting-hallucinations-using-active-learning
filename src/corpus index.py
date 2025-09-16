# src/corpus_index.py
from __future__ import annotations
import os, json
from typing import List, Dict, Any, Optional
import numpy as np

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

    def search(self, query_vec: np.ndarray, k: int = 5):
        q = query_vec.reshape(1, -1).astype(np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        if self.index is not None:
            q2 = q.copy()
            faiss.normalize_L2(q2)
            sims, idxs = self.index.search(q2, k)
            return sims[0], idxs[0]
        # NumPy fallback
        sims_full = (q @ self.emb.T).ravel()
        if k >= sims_full.size:
            top_idx = np.argsort(-sims_full)
        else:
            part = np.argpartition(-sims_full, k)[:k]
            top_idx = part[np.argsort(-sims_full[part])]
        return sims_full[top_idx], top_idx

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
